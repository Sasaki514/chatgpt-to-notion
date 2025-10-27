import os
import re
import time
import json
import glob
import zipfile
import argparse
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import requests
from dotenv import load_dotenv

# OpenAI（要約を使う場合のみ）
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # OPENAIを使わない運用でも動く

# プロンプトと週報機能をdaily_chatgpt_summaryからインポート
from daily_chatgpt_summary import (
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    should_create_weekly_report, get_weekly_date_range,
    create_weekly_report, get_last_weekly_report_date,
    save_last_weekly_report_date, has_sufficient_weekly_data,
    jst_today
)

JST = timezone(timedelta(hours=9))
STATE_FILE = "state.json"


def jst_now_iso(): return datetime.now(JST).isoformat()
def jst_today(): return datetime.now(JST).date().isoformat()

# ---------- state.json（前回以降のみ処理する"しおり"） ----------


def load_state(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "conv_hwm": {}, "seen": {}}


def save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# ---------- ZIPファイル検索 ----------


def get_downloads_dir():
    d = os.getenv("DOWNLOADS_DIR")
    if d:
        return d
    return os.path.join(os.path.expanduser("~"), "Downloads")

# ---------- ZIP → conversations.json ----------


def unzip_to_tmp(zip_path, workdir):
    tmp = os.path.join(workdir, "_tmp_unzip")
    if os.path.exists(tmp):
        for f in glob.glob(os.path.join(tmp, "*")):
            try:
                if os.path.isdir(f):
                    import shutil
                    shutil.rmtree(f, ignore_errors=True)
                else:
                    os.remove(f)
            except Exception:
                pass
    os.makedirs(tmp, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    return os.path.join(tmp, "conversations.json")


def load_conversations(path):
    if not os.path.exists(path):
        raise FileNotFoundError("conversations.json が見つかりません（エクスポート内容を確認）。")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "conversations" in data:
        return data["conversations"]
    return data

# ---------- エクスポート構造の読み出し ----------


def iter_messages(conv):
    """
    mapping 構造を想定。無ければ messages 配列にフォールバック。
    return: (conv_id, role, text, ts(UTC秒/None), mid, title)
    """
    conv_id = conv.get("id") or conv.get("conversation_id") or "(unknown)"
    title = conv.get("title") or ""
    mapping = conv.get("mapping")
    if mapping:
        for node_id, node in mapping.items():
            msg = (node.get("message") or {})
            role = (msg.get("author") or {}).get("role")
            content = msg.get("content")
            if not role or not content:
                continue
            text = None
            if isinstance(content, dict) and "parts" in content:
                parts = content.get("parts") or []
                text = "\n".join(p for p in parts if isinstance(p, str))
            elif isinstance(content, str):
                text = content
            if text:
                ts = msg.get("create_time")
                mid = msg.get("id") or f"mapping-{node_id}"
                yield conv_id, role, text, ts, mid, title
    else:
        # フォールバック: messages配列
        messages = conv.get("messages", [])
        for m in messages:
            role = m.get("author", {}).get("role")
            content = m.get("content")
            text = None
            if isinstance(content, dict) and "parts" in content:
                parts = content.get("parts") or []
                text = "\n".join(p for p in parts if isinstance(p, str))
            elif isinstance(content, str):
                text = content
            ts = m.get("create_time")
            if role and text:
                mid = m.get("id") or f"legacy-{hash((role,text,ts))}"
                yield conv_id, role, text, ts, mid, title


def ts_to_day(ts, from_date_str="2025-09-25"):
    if ts is None:
        return jst_today()
    try:
        date_obj = datetime.fromtimestamp(
            ts, tz=timezone.utc).astimezone(JST).date()
        from_date = datetime.strptime(from_date_str, "%Y-%m-%d").date()
        # 指定日以降の日付のみを処理
        if date_obj >= from_date:
            return date_obj.isoformat()
        else:
            return None  # 指定日以前は除外
    except Exception:
        return jst_today()


def build_daily_raw(conversations, state, max_chars=16000, from_date_str="2025-09-25"):
    """
    state（conv_hwm/seen）を使って"前回以降のみ"を日付ごとにまとめる
    クリーンアップ対応: タイムスタンプベースの判定を優先
    return: (day -> raw_text, progress: conv_id -> (new_seen_ids, max_ts))
    """
    hwm = state["conv_hwm"]
    seen = state["seen"]

    # クリーンアップ後のstateかどうかを判定
    is_optimized_state = state.get("version", 1) >= 2
    if is_optimized_state:
        print(f"[INFO] 最適化されたstate.jsonを使用中（バージョン: {state.get('version', 1)}）")
        print(f"[INFO] タイムスタンプベースの判定を優先します")
    buckets = defaultdict(list)
    progress = {}

    for conv in conversations:
        last_ts_map = hwm
        conv_id = conv.get("id") or conv.get("conversation_id") or "(unknown)"
        last_ts = last_ts_map.get(conv_id, -1)
        seen_set = set(seen.get(conv_id, []))
        new_seen = set()
        max_ts = last_ts

        for cid, role, text, ts, mid, title in iter_messages(conv):
            # タイムスタンプベースの判定を優先（クリーンアップ対応）
            if ts is not None and ts <= last_ts:
                continue

            # メッセージIDベースの判定（重複回避）
            if mid in seen_set:
                continue
            day = ts_to_day(ts, from_date_str)
            if day is None:
                # 指定日以前の日付はスキップ
                continue
            prefix = "ユーザー" if role == "user" else (
                "アシスタント" if role == "assistant" else role)
            buckets[day].append(f"[{prefix}] {title}｜{text.strip()}")
            new_seen.add(mid)
            if ts is not None and ts > max_ts:
                max_ts = ts

        if new_seen:
            progress[conv_id] = (new_seen, max_ts)

    daily = {}
    for day, lines in buckets.items():
        joined = "\n- " + "\n- ".join(lines)
        if len(joined) > max_chars:
            joined = joined[:max_chars] + "\n…（長文のため途中まで）"
        daily[day] = joined
    return daily, progress


def build_daily_raw_all_data(conversations, max_chars=16000, from_date_str="2025-09-25"):
    """
    テスト用: 全てのデータを処理（差分処理なし）
    return: (day -> raw_text, progress: conv_id -> (new_seen_ids, max_ts))
    """
    buckets = defaultdict(list)
    progress = {}

    for conv in conversations:
        conv_id = conv.get("id") or conv.get("conversation_id") or "(unknown)"
        seen_set = set()  # 空のセット（全て処理）
        new_seen = set()
        max_ts = 0

        for cid, role, text, ts, mid, title in iter_messages(conv):
            # 全てのメッセージを処理（差分チェックなし）
            day = ts_to_day(ts, from_date_str)
            if day is None:
                # 指定日以前の日付はスキップ
                continue
            prefix = "ユーザー" if role == "user" else (
                "アシスタント" if role == "assistant" else role)
            buckets[day].append(f"[{prefix}] {title}｜{text.strip()}")
            new_seen.add(mid)
            if ts is not None and ts > max_ts:
                max_ts = ts

        if new_seen:
            progress[conv_id] = (new_seen, max_ts)

    daily = {}
    for day, lines in buckets.items():
        joined = "\n- " + "\n- ".join(lines)
        if len(joined) > max_chars:
            joined = joined[:max_chars] + "\n…（長文のため途中まで）"
        daily[day] = joined
    return daily, progress


# ---------- 要約（任意） ----------
PROMPT = """以下のChatGPT相談ログを分析し、日付ごとに要約してください。

重要な指示：
1. 生のログをそのまま出力してはいけません
2. 必ず以下の形式で要約してください
3. 各日の相談内容を3-5個のトピックに整理してください

出力形式（厳守）：
## 日付（YYYY-MM-DD）

### 〔相談したトピック名〕
**要点:** 相談の要点を簡潔にまとめ
**次のアクション:** 今後の調査・学習すべき内容

（以下、他の日付についても同様の形式で続ける）

相談ログ：
{notes}
"""


def summarize(raw_text, api_key, model):
    print(f"[DEBUG] ===== summarize関数呼び出し =====")
    print(f"[DEBUG] OpenAI: {'利用可能' if OpenAI else '利用不可'}")
    print(f"[DEBUG] モデル: {model}")
    print(f"[DEBUG] 入力テキスト長: {len(raw_text)}文字")

    if not api_key or not OpenAI:
        # 要約しない（生ログを保存）
        print(f"[DEBUG] OpenAIが未設定のため、生ログを返します")
        day = jst_today()
        return f"## 📅 {day} ChatGPT振り返り（生ログ）\n\n```\n{raw_text}\n```"

    # インポートしたプロンプトを使用
    user_prompt = USER_PROMPT_TEMPLATE.format(raw_text=raw_text)

    print(f"📤 ChatGPT API投げました:")
    print(f"   モデル: {model}")
    print(f"   入力テキスト長: {len(raw_text)}文字")
    print(f"   入力データの先頭: {raw_text[:300]}...")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3  # より一貫した出力のため
    )

    result = resp.choices[0].message.content.strip()

    print(f"📥 ChatGPTのレスポンス:")
    print(f"   出力長: {len(result)}文字")
    print(
        f"   使用トークン: {resp.usage.total_tokens if hasattr(resp, 'usage') else '不明'}")
    print(f"   '##'の数: {result.count('##')}")
    print(f"   '###'の数: {result.count('###')}")
    print(f"   内容: {result[:300]}..." if len(
        result) > 300 else f"   内容: {result}")

    # トピック間の改行を整形
    # ### の前に改行がない場合は追加
    result = re.sub(r'([^\n])\n(###\s)', r'\1\n\n\2', result)
    # ### の直後に改行が1つしかない場合も調整
    result = re.sub(r'(###[^\n]+)\n([^\n])', r'\1\n\n\2', result)

    print(f"📝 整形後:")
    print(f"   出力長: {len(result)}文字")

    return result

# ---------- Notion ----------


def markdown_to_notion_blocks(markdown_text, max_chars_per_block=1900):
    """
    MarkdownテキストをNotionブロック形式に変換（太字記法対応）
    """
    import re

    def parse_rich_text(text):
        """テキストを解析してNotionのrich_text形式に変換"""
        rich_text = []
        current_pos = 0

        # **太字** パターンを検索
        bold_pattern = r'\*\*(.*?)\*\*'
        for match in re.finditer(bold_pattern, text):
            # 太字の前のテキスト
            if match.start() > current_pos:
                normal_text = text[current_pos:match.start()]
                if normal_text:
                    rich_text.append({
                        "type": "text",
                        "text": {"content": normal_text}
                    })

            # 太字部分
            bold_text = match.group(1)
            rich_text.append({
                "type": "text",
                "text": {"content": bold_text},
                "annotations": {"bold": True}
            })

            current_pos = match.end()

        # 残りのテキスト
        if current_pos < len(text):
            remaining_text = text[current_pos:]
            if remaining_text:
                rich_text.append({
                    "type": "text",
                    "text": {"content": remaining_text}
                })

        return rich_text if rich_text else [{"type": "text", "text": {"content": text}}]

    blocks = []
    lines = markdown_text.split('\n')
    current_content = ""

    for line_num, line in enumerate(lines, 1):

        # 見出しの処理（厳密な条件で判定）
        is_heading3 = line.startswith('###') and not line.startswith('####')
        is_heading2 = line.startswith('##') and not line.startswith('###')
        is_heading1 = line.startswith('#') and not line.startswith(
            '##') and not line.startswith('###')

        if is_heading3:
            # 現在のコンテンツがある場合は保存
            if current_content.strip():
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": parse_rich_text(current_content.strip())
                    }
                })
                current_content = ""

            # 小見出しブロックを追加（###の後の空白を適切に処理、空の見出しは無視）
            heading_text = line[3:].strip()  # "###" を除去してからstrip
            if heading_text.startswith(' '):
                heading_text = heading_text[1:]  # 先頭の空白を除去

            # 空の見出しは作成しない
            if heading_text:
                blocks.append({
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": parse_rich_text(heading_text)
                    }
                })
                # トピック間の行間を空けるために空の段落を2つ追加
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": ""}}]
                    }
                })
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": ""}}]
                    }
                })
        elif is_heading2:
            # 現在のコンテンツがある場合は保存
            if current_content.strip():
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": parse_rich_text(current_content.strip())
                    }
                })
                current_content = ""

            # 見出しブロックを追加（空の見出しは無視）
            heading_text = line[2:].strip()  # "##" を除去してからstrip
            if heading_text.startswith(' '):
                heading_text = heading_text[1:]  # 先頭の空白を除去

            # 空の見出しは作成しない
            if heading_text:
                blocks.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": parse_rich_text(heading_text)
                    }
                })
                # トピック間の行間を空けるために空の段落を2つ追加
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": ""}}]
                    }
                })
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": ""}}]
                    }
                })
        elif is_heading1:
            # 現在のコンテンツがある場合は保存
            if current_content.strip():
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": parse_rich_text(current_content.strip())
                    }
                })
                current_content = ""

            # 大見出しブロックを追加（空の見出しは無視）
            heading_text = line[1:].strip()  # "#" を除去してからstrip
            if heading_text.startswith(' '):
                heading_text = heading_text[1:]  # 先頭の空白を除去

            # 空の見出しは作成しない
            if heading_text:
                blocks.append({
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": parse_rich_text(heading_text)
                    }
                })
                # トピック間の行間を空けるために空の段落を2つ追加
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": ""}}]
                    }
                })
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": ""}}]
                    }
                })
        else:
            # 通常のテキスト（空行は無視）
            if line.strip():  # 空行でない場合のみ追加
                # 「次のアクション」の行の場合は追加の改行を入れる
                if line.strip().startswith('**次のアクション:**'):
                    current_content += line + '\n\n'  # 追加の改行
                else:
                    current_content += line + '\n'

            # 文字数制限チェック
            if len(current_content) > max_chars_per_block:
                # 現在のコンテンツをブロックとして保存
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": parse_rich_text(current_content.strip())
                    }
                })
                current_content = ""

    # 残りのコンテンツがある場合は保存
    if current_content.strip():
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": parse_rich_text(current_content.strip())
            }
        })

    return blocks


def notion_get_title_date_props(token, dbid):
    h = {"Authorization": f"Bearer {token}", "Notion-Version": "2022-06-28"}
    r = requests.get(f"https://api.notion.com/v1/databases/{dbid}", headers=h)
    r.raise_for_status()
    data = r.json()
    title_prop = next(
        k for k, v in data["properties"].items() if v["type"] == "title")
    date_prop = next(
        (k for k, v in data["properties"].items() if v["type"] == "date"), None)
    return title_prop, date_prop


def notion_create_page(token, dbid, title_prop, date_prop, day, markdown):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }
    props = {title_prop: {
        "title": [{"text": {"content": f"{day} ChatGPT振り返り"}}]}}
    if date_prop:
        # 3日後の12:00〜13:00（JST）を設定
        date_obj = datetime.strptime(day, "%Y-%m-%d").date()
        next_day = date_obj + timedelta(days=3)

        # UTCで日本時間12-13時を指定（UTC = JST - 9時間）
        start_time_utc = datetime.combine(next_day, datetime.min.time().replace(
            hour=12, minute=0)).replace(tzinfo=timezone.utc)  # JST 12:00 = UTC 03:00
        end_time_utc = datetime.combine(next_day, datetime.min.time().replace(
            hour=13, minute=0)).replace(tzinfo=timezone.utc)  # JST 13:00 = UTC 04:00

        props[date_prop] = {
            "date": {
                "start": start_time_utc.isoformat(),
                "end": end_time_utc.isoformat(),
                "time_zone": "Asia/Tokyo"
            }
        }
    # MarkdownをNotionブロック形式に変換
    children = markdown_to_notion_blocks(markdown)

    # デバッグ用: 変換結果を表示
    print(f"[DEBUG] 変換されたブロック数: {len(children)}")
    print(f"[DEBUG] 元のMarkdownテキスト: {repr(markdown[:200])}")
    for i, block in enumerate(children[:5]):  # 最初の5ブロックを表示
        block_type = block.get('type', 'unknown')
        if block_type == 'heading_3':
            content = block.get('heading_3', {}).get('rich_text', [{}])[
                0].get('text', {}).get('content', '')
            print(f"[DEBUG] ブロック{i+1}: {block_type} - 内容: {repr(content)}")
        elif block_type == 'paragraph':
            content = block.get('paragraph', {}).get('rich_text', [{}])[
                0].get('text', {}).get('content', '')
            print(
                f"[DEBUG] ブロック{i+1}: {block_type} - 内容: {repr(content[:50])}")
        else:
            print(f"[DEBUG] ブロック{i+1}: {block_type} - {str(block)[:100]}...")

    payload = {"parent": {"database_id": dbid},
               "properties": props, "children": children}

    # データ検証
    print(f"[DEBUG] ===== データ検証 =====")
    print(f"[DEBUG] プロパティ: {props}")
    print(f"[DEBUG] 子ブロック数: {len(children)}")

    # プロパティの検証
    if not props.get(title_prop):
        print(f"[ERROR] タイトルプロパティ '{title_prop}' が見つかりません")
        return None

    # 子ブロックの検証
    if not children:
        print(f"[WARNING] 子ブロックが空です")
    else:
        print(f"[DEBUG] 子ブロック詳細:")
        for i, child in enumerate(children[:5]):  # 最初の5つだけ表示
            block_type = child.get('type', 'unknown')
            if block_type == 'heading_1':
                content = child.get('heading_1', {}).get('rich_text', [{}])[
                    0].get('text', {}).get('content', '')
                print(f"[DEBUG]   ブロック{i+1}: heading_1 - '{content}'")
            elif block_type == 'heading_2':
                content = child.get('heading_2', {}).get('rich_text', [{}])[
                    0].get('text', {}).get('content', '')
                print(f"[DEBUG]   ブロック{i+1}: heading_2 - '{content}'")
            elif block_type == 'heading_3':
                content = child.get('heading_3', {}).get('rich_text', [{}])[
                    0].get('text', {}).get('content', '')
                print(f"[DEBUG]   ブロック{i+1}: heading_3 - '{content}'")
            elif block_type == 'paragraph':
                content = child.get('paragraph', {}).get('rich_text', [{}])[
                    0].get('text', {}).get('content', '')
                print(
                    f"[DEBUG]   ブロック{i+1}: paragraph - '{content[:100]}{'...' if len(content) > 100 else ''}'")
            else:
                print(
                    f"[DEBUG]   ブロック{i+1}: {block_type} - {str(child)[:100]}...")

    # ペイロードサイズの検証
    payload_size = len(str(payload))
    print(f"[DEBUG] ペイロードサイズ: {payload_size}文字")
    if payload_size > 100000:  # 100KB制限
        print(f"[WARNING] ペイロードサイズが大きすぎる可能性があります")

    print(f"[DEBUG] ========================")

    r = requests.post("https://api.notion.com/v1/pages",
                      headers=headers, json=payload)

    # エラーレスポンスの詳細を表示
    if r.status_code != 200:
        print(f"[ERROR] Notion API エラー: {r.status_code}")
        print(f"[ERROR] レスポンス: {r.text}")
        try:
            error_detail = r.json()
            print(f"[ERROR] 詳細: {error_detail}")
        except:
            pass

        # エラー詳細をログファイルにも記録
        error_log = f"Notion API Error {r.status_code}: {r.text}"
        print(f"[ERROR] エラーログ: {error_log}")

        # 400エラーの場合、ペイロードの詳細も表示
        if r.status_code == 400:
            print(f"[ERROR] 送信されたペイロード:")
            print(f"[ERROR] プロパティ: {props}")
            print(f"[ERROR] 子ブロック数: {len(children)}")
            print(f"[ERROR] 最初の3つの子ブロック:")
            for i, child in enumerate(children[:3]):
                print(f"[ERROR]   ブロック{i+1}: {child}")

        # エラーを再発生させずにNoneを返す
        return None

    r.raise_for_status()
    return r.json().get("url")


def notion_create_weekly_page(token, dbid, title_prop, date_prop, title, date_with_time, markdown):
    """週報用のNotionページ作成"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }
    props = {title_prop: {
        "title": [{"text": {"content": title}}]}}
    if date_prop:
        props[date_prop] = {"date": date_with_time}

    # MarkdownをNotionブロック形式に変換
    children = markdown_to_notion_blocks(markdown)

    payload = {"parent": {"database_id": dbid},
               "properties": props, "children": children}

    r = requests.post("https://api.notion.com/v1/pages", headers=headers,
                      data=json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    if r.status_code not in (200, 201):
        raise RuntimeError(f"週報ページ作成に失敗: {r.status_code} {r.text}")
    return r.json()

# ---------- 週間統計取得 ----------


def get_weekly_conversations_with_stats(conversations, monday, friday):
    """指定期間の会話ログと統計情報を取得"""
    from datetime import datetime, timezone

    # 日付範囲をdatetimeオブジェクトに変換
    monday_dt = datetime.strptime(
        monday, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    friday_dt = datetime.strptime(
        friday, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # 統計情報の初期化
    stats = {
        'conversation_count': 0,
        'user_message_count': 0,
        'assistant_message_count': 0,
        'total_duration_minutes': 0.0
    }

    weekly_conversations = []

    for conv in conversations:
        # 会話IDを取得
        conv_id = conv.get("id") or conv.get("conversation_id") or "(unknown)"

        # 会話の作成日時を確認
        create_time = conv['create_time']
        if isinstance(create_time, float):
            # タイムスタンプ形式の場合
            created_time = datetime.fromtimestamp(create_time, tz=timezone.utc)
        elif isinstance(create_time, str):
            # ISO文字列形式の場合
            created_time = datetime.fromisoformat(
                create_time.replace('Z', '+00:00'))
        else:
            continue  # 不明な形式の場合はスキップ

        # 指定期間内の会話のみ処理
        if monday_dt <= created_time <= friday_dt:
            stats['conversation_count'] += 1

            # メッセージ数をカウント
            for message in conv.get('mapping', {}).values():
                if message.get('message'):
                    author = message['message'].get(
                        'author', {}).get('role', '')
                    if author == 'user':
                        stats['user_message_count'] += 1
                    elif author == 'assistant':
                        stats['assistant_message_count'] += 1

            # 会話時間を計算（最初と最後のメッセージの時間差）
            message_times = []
            for message in conv.get('mapping', {}).values():
                if message.get('message') and message['message'].get('create_time'):
                    msg_create_time = message['message']['create_time']
                    if isinstance(msg_create_time, float):
                        # タイムスタンプ形式の場合
                        msg_time = datetime.fromtimestamp(
                            msg_create_time, tz=timezone.utc)
                    elif isinstance(msg_create_time, str):
                        # ISO文字列形式の場合
                        msg_time = datetime.fromisoformat(
                            msg_create_time.replace('Z', '+00:00'))
                    else:
                        continue  # 不明な形式の場合はスキップ
                    message_times.append(msg_time)

            if len(message_times) >= 2:
                duration = (max(message_times) - min(message_times)
                            ).total_seconds() / 60
                stats['total_duration_minutes'] += duration

            # 会話内容をテキスト形式で保存
            conv_text = f"=== 会話 {conv_id} ({created_time.strftime('%Y-%m-%d %H:%M')}) ===\n"
            for message in conv.get('mapping', {}).values():
                if message.get('message'):
                    author = message['message'].get(
                        'author', {}).get('role', '')
                    content = message['message'].get(
                        'content', {}).get('parts', [''])[0]
                    if author == 'user':
                        conv_text += f"[ユーザー] {content}\n"
                    elif author == 'assistant':
                        conv_text += f"[アシスタント] {content}\n"
            conv_text += "\n"

            weekly_conversations.append(conv_text)

    # 週間ログテキストを結合
    weekly_raw_text = f"=== 週間相談ログ ({monday} 〜 {friday}) ===\n\n"
    weekly_raw_text += f"📊 週間統計:\n"
    weekly_raw_text += f"- チャット会話数: {stats['conversation_count']}回\n"
    weekly_raw_text += f"- ユーザーメッセージ数: {stats['user_message_count']}回\n"
    weekly_raw_text += f"- アシスタントメッセージ数: {stats['assistant_message_count']}回\n"
    weekly_raw_text += f"- 総会話時間: {stats['total_duration_minutes']:.1f}分\n\n"
    weekly_raw_text += "".join(weekly_conversations)

    return weekly_raw_text, stats

# ---------- メイン ----------


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="ZIPファイルから差分集約→Notion登録")
    parser.add_argument("zip_file", nargs="?", default=None,
                        help="ChatGPTエクスポートZIPファイルのパス（未指定時は自動検索）")
    parser.add_argument("--workdir", default=os.getenv("WORK_DIR", "./ChatGPT_Notion"),
                        help="作業ディレクトリ（state.jsonなど）")
    parser.add_argument("--from-date", default="2025-09-18",
                        help="処理する日付の開始日（YYYY-MM-DD、既定: 2025-09-18）")
    args = parser.parse_args()

    notion_token = os.getenv("NOTION_TOKEN")
    notion_dbid = os.getenv("DATABASE_ID")
    if not notion_token or not notion_dbid:
        raise SystemExit("NOTION_TOKEN / DATABASE_ID が未設定です（.env）。")

    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # デバッグ用: 環境変数の読み込み状況を表示
    print(f"[DEBUG] ===== 環境変数確認 =====")
    print(f"[DEBUG] 環境変数読み込み完了")
    print(f"[DEBUG] ========================")

    os.makedirs(args.workdir, exist_ok=True)
    state_path = os.path.join(args.workdir, STATE_FILE)
    state = load_state(state_path)

    # 自動クリーンアップ機能（オプション）
    try:
        from state_cleanup import auto_cleanup_state
        state = auto_cleanup_state(state_path, state)
    except ImportError:
        print("[INFO] state_cleanupモジュールが見つかりません。クリーンアップをスキップします。")
    except Exception as e:
        print(f"[WARNING] 自動クリーンアップ中にエラー: {e}")

    # --- ZIPファイルの取得 ---
    if args.zip_file:
        zip_path = args.zip_file
        if not os.path.exists(zip_path):
            raise SystemExit(f"指定されたZIPファイルが見つかりません: {zip_path}")
        print(f"[OK] 指定ZIP: {zip_path}")
    else:
        # 自動検索: Downloadsディレクトリから最新のZIPファイルを探す
        downloads = get_downloads_dir()
        print(f"[INFO] Downloadsディレクトリを検索中: {downloads}")

        # 全てのZIPファイルを検索（ファイル名の規則性に関係なく）
        zip_files = glob.glob(os.path.join(downloads, "*.zip"))

        if not zip_files:
            raise SystemExit(
                "ZIPファイルが見つかりません。Downloadsディレクトリを確認するか、--zip-file で直接指定してください。")

        # 最新のファイルを選択（更新日時でソート）
        zip_path = max(zip_files, key=os.path.getmtime)
        print(f"[OK] 最新ZIP: {zip_path}")

    # --- ZIP → conversations.json 読み込み ---
    conv_json = unzip_to_tmp(zip_path, args.workdir)
    conversations = load_conversations(conv_json)

    # 差分処理（stateと比較して新規メッセージのみ処理）
    daily_raw, progress = build_daily_raw(
        conversations, state, from_date_str=args.from_date)

    if not daily_raw:
        print("[INFO] 前回以降の新規メッセージなし。")
        print("週報作成判断を実行します...")
        # 日報処理はスキップ、週報作成判断のみ実行
    else:
        print(f"[INFO] {len(daily_raw)}日分の新規メッセージを処理します。")

    # --- Notionプロパティ ---
    title_prop, date_prop = notion_get_title_date_props(
        notion_token, notion_dbid)

    # --- 日報処理（新規メッセージがある場合のみ） ---
    if daily_raw:
        print("\n=== 日報処理 ===")
    # --- 複数日をまとめて要約→Notion登録（コンテキストウィンドウ制限対応） ---
    MAX_CHARS_PER_REQUEST = 120000  # 安全マージンを持たせた制限

    # 全日の内容を結合
    print(f"[DEBUG] daily_rawの内容: {daily_raw}")
    print(f"[DEBUG] daily_rawのキー数: {len(daily_raw)}")

    combined_text = ""
    for day in sorted(daily_raw.keys()):
        combined_text += f"\n## {day}\n{daily_raw[day]}\n"

    print(f"[DEBUG] combined_textの長さ: {len(combined_text)}")
    print(f"[DEBUG] combined_textの先頭: {combined_text[:200]}...")

    # combined_textが空の場合はAPI呼び出しをスキップ
    if len(combined_text.strip()) == 0:
        print("[INFO] 処理対象のテキストが空のため、API呼び出しをスキップします。")
        daily_summaries = {}
    elif len(combined_text) > MAX_CHARS_PER_REQUEST:
        # テキストが長すぎる場合は分割
        print(f"[WARN] 全日の内容が長すぎるため分割して処理します...")
        # 長いテキストを分割
        chunks = []
        current_chunk = ""
        lines = combined_text.split('\n')

        for line in lines:
            if len(current_chunk + line) > MAX_CHARS_PER_REQUEST and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # 各チャンクを要約して結合
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"チャンク {i+1}/{len(chunks)} を要約中...")
            chunk_md = summarize(chunk, openai_key, openai_model)
            chunk_summaries.append(f"## 部分 {i+1}\n{chunk_md}")

        # チャンクの要約を結合
        combined_md = f"## 複数日 ChatGPT振り返り（分割処理）\n\n" + \
            "\n\n".join(chunk_summaries)
    else:
        # 1回のAPIで要約
        print(f"[INFO] {len(daily_raw)}日分をまとめて要約中（OpenAI: {openai_model}）...")
        combined_md = summarize(combined_text, openai_key, openai_model)

    # API呼び出し後の処理
    if 'combined_md' in locals():
        # 要約結果を日付ごとに分割
        print(f"📋 要約結果の分割処理:")
        print(f"   要約結果全体の長さ: {len(combined_md)}文字")
        print(f"   要約結果の先頭: {combined_md[:500]}...")

        # 日付の見出し（## YYYY-MM-DD）で分割（### は除外）
        import re
        sections = re.split(r'\n(?=## \d{4}-\d{2}-\d{2})', combined_md)
        daily_summaries = {}

        print(f"   分割されたセクション数: {len(sections)}")

        for i, section in enumerate(sections):
            print(f"   セクション{i}: 長さ={len(section)}, 内容='{section[:100]}...'")

        print(f"📋 日付ごとに要約結果まとめました:")

        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            lines = section.split('\n')

            # 最初の行から日付を抽出（## を除去）
            first_line = lines[0].strip()
            if first_line.startswith('## '):
                day = first_line[3:].strip()  # "## " を除去
            elif re.match(r'^\d{4}-\d{2}-\d{2}', first_line):
                # "## " がない場合でも日付形式なら受け入れる
                day = first_line
            else:
                print(f"⚠️ セクション{i}の最初の行が日付形式ではありません: {first_line}")
                continue

            # 日付の検証
            try:
                datetime.strptime(day, "%Y-%m-%d")
            except ValueError:
                print(f"⚠️ セクション{i}の日付形式が不正です: {day}")
                continue

            # 内容を取得（2行目以降）
            content = '\n'.join(lines[1:]).strip()

            print(f"   {day}: {len(content)}文字")
            daily_summaries[day] = content
    else:
        print("[INFO] API呼び出しがスキップされたため、要約処理もスキップします。")
        daily_summaries = {}

    # 各日をNotionに登録
    print(f"📝 Notion登録時の内容:")
    print(f"   利用可能な要約結果: {list(daily_summaries.keys())}")
    print(f"   処理対象の日付: {list(daily_raw.keys())}")

    # 成功した日付を追跡
    successful_days = []
    failed_days = []

    for day in sorted(daily_raw.keys()):
        print(f"   処理中の日付: '{day}'")
        print(f"   daily_summariesに存在するか: {day in daily_summaries}")

        # 要約があるかチェック
        if day in daily_summaries:
            summary_content = daily_summaries[day]
            print(f"   要約内容の長さ: {len(summary_content)}")
            print(f"   要約内容: '{summary_content}'")

            # 要約内容が実質的に空でないかチェック
            if summary_content.strip() and len(summary_content.strip()) > 5:
                md = f"## {day}\n{summary_content}"  # 要約結果を使用
                data_type = "要約結果"
                print(f"   → 要約結果を使用")
            else:
                md = f"## {day}\n\n会話内容なし（要約が空）"  # 要約が空の場合
                data_type = "会話内容なし（要約が空）"
                print(f"   → 要約が空のため「会話内容なし」を使用")
        else:
            md = f"## {day}\n\n会話内容なし（該当日付のデータなし）"  # 該当日付なし
            data_type = "会話内容なし（データなし）"
            print(f"   → 該当日付のデータがないため「会話内容なし」を使用")

        print(f"   最終的な使用データ: {data_type} ({len(md)}文字)")

        url = notion_create_page(
            notion_token, notion_dbid, title_prop, date_prop, day, md)

        if url:
            print(f"   ✅ Notion保存完了: {day}")
            print(f"   📄 ページURL: {url}")
            successful_days.append(day)
        else:
            print(f"   ❌ Notion保存失敗: {day}")
            print(f"   ⚠️ エラーの詳細は上記のログを確認してください")
            failed_days.append(day)

            # エラーが発生した場合、スクリプトを停止
            print(f"\n🚨 エラー発生により処理を停止します")
            print(f"   成功した日付: {successful_days}")
            print(f"   失敗した日付: {failed_days}")
            print(f"   state.jsonは更新されません")
            raise SystemExit(f"Notion API エラーにより処理を停止しました。日付: {day}")

    # すべて成功した場合のみstate更新
    if successful_days and not failed_days:
        print(f"\n✅ すべての日付でNotion保存が成功しました")
        for conv_id, (new_seen, max_ts) in progress.items():
            state["seen"].setdefault(conv_id, [])
            state["seen"][conv_id].extend(list(new_seen))
            if max_ts > state["conv_hwm"].get(conv_id, -1):
                state["conv_hwm"][conv_id] = max_ts
        save_state(state_path, state)
        print(f"[DONE] 日報登録完了。state更新済み。")
    else:
        print("[INFO] 日報処理はスキップされました。")

    # 週報作成チェック
    print("\n=== 週報作成チェック ===")
    workdir = os.getenv("WORK_DIR", "./ChatGPT_Notion")

    # 週報作成判断の詳細ログ
    print("🔍 週報作成判断プロセス:")
    last_weekly_date = get_last_weekly_report_date(workdir)
    # 最初に読み取った値を保持（表示用）
    original_last_weekly_date = last_weekly_date
    print(
        f"   最終登録日: {original_last_weekly_date if original_last_weekly_date else '未登録'}")

    today = jst_today()
    today_date = datetime.strptime(today, "%Y-%m-%d").date()
    print(f"   今日の日付: {today}")

    if last_weekly_date:
        last_date = datetime.strptime(last_weekly_date, "%Y-%m-%d").date()
        last_date_weekday = last_date.weekday()
        next_saturday = last_date - \
            timedelta(days=last_date_weekday) + timedelta(days=5)
        if next_saturday <= last_date:
            next_saturday += timedelta(days=7)
        print(f"   前回登録日: {last_date}")
        print(f"   次の土曜日: {next_saturday}")
        print(f"   今日 >= 次の土曜: {today_date >= next_saturday}")

        should_create = today_date >= next_saturday
    else:
        print("   未登録のため、前の金曜日までのデータで週報作成を検討")
        should_create = has_sufficient_weekly_data(workdir)

    print(f"   週報作成判定: {should_create}")

    if should_create:
        print("📅 週報作成を実行します")

        # 今週の日付範囲を取得
        monday, friday = get_weekly_date_range()
        print(f"   対象期間: {monday} 〜 {friday}")

        # 週間の相談ログを取得（実際のログから統計情報付きで取得）
        weekly_raw_text, weekly_stats = get_weekly_conversations_with_stats(
            conversations, monday, friday)

        print(f"   週間統計:")
        print(f"     チャット会話数: {weekly_stats['conversation_count']}回")
        print(f"     ユーザーメッセージ数: {weekly_stats['user_message_count']}回")
        print(f"     アシスタントメッセージ数: {weekly_stats['assistant_message_count']}回")
        print(f"     会話時間: {weekly_stats['total_duration_minutes']:.1f}分")

        # 週報作成
        weekly_report = create_weekly_report(
            weekly_raw_text, openai_key, openai_model)

        if weekly_report:
            # 週報をNotionに登録
            today = jst_today()
            today_date = datetime.strptime(today, "%Y-%m-%d").date()

            # 登録日付の決定
            if original_last_weekly_date:
                # 条件2: 最新登録日付の次の土曜日
                last_date = datetime.strptime(
                    original_last_weekly_date, "%Y-%m-%d").date()
                last_date_weekday = last_date.weekday()
                next_saturday = last_date - \
                    timedelta(days=last_date_weekday) + timedelta(days=5)
                if next_saturday <= last_date:
                    next_saturday += timedelta(days=7)
                # 土曜日を計算してから日曜日に変更
                registration_date = next_saturday + timedelta(days=1)
                print(f"   登録日付: {registration_date} (前回登録日{last_date}の次の日曜日)")
            else:
                # 条件1: システム日付直近の金曜日（システム日付が金曜の場合は登録しない）
                days_since_monday = today_date.weekday()
                monday = today_date - timedelta(days=days_since_monday)
                friday = monday + timedelta(days=4)

                if today_date == friday:
                    print("   今日が金曜日のため、週報登録をスキップします")
                    return

                # 金曜日を計算してから日曜日に変更
                registration_date = friday + timedelta(days=2)
                print(f"   登録日付: {registration_date} (直近の日曜日)")

            weekly_title = f"{registration_date} 週間学習レポート"

            # UTCで日本時間12-13時を指定（UTC = JST - 9時間）
            start_time_utc = datetime.combine(
                registration_date, datetime.min.time().replace(hour=3, minute=0)).replace(tzinfo=timezone.utc)  # JST 12:00 = UTC 03:00
            end_time_utc = datetime.combine(
                registration_date, datetime.min.time().replace(hour=4, minute=0)).replace(tzinfo=timezone.utc)  # JST 13:00 = UTC 04:00

            weekly_date_with_time = {
                "start": start_time_utc.isoformat(),
                "end": end_time_utc.isoformat(),
                "time_zone": "Asia/Tokyo"
            }

            try:
                weekly_page = notion_create_weekly_page(notion_token, notion_dbid, title_prop,
                                                        date_prop, weekly_title, weekly_date_with_time, weekly_report)

                if weekly_page and weekly_page.get("url"):
                    # 最終週報登録日を保存（登録日付を使用）
                    save_last_weekly_report_date(
                        workdir, registration_date.isoformat())

                    print("✅ 週報保存完了")
                    print("週報ページURL:", weekly_page.get("url"))
                else:
                    print("❌ 週報作成に失敗しました")
                    print("⚠️ エラーの詳細は上記のログを確認してください")
                    raise SystemExit("週報作成エラーにより処理を停止しました")
            except Exception as e:
                print(f"❌ 週報作成中にエラーが発生しました: {e}")
                raise SystemExit(f"週報作成エラーにより処理を停止しました: {e}")
        else:
            print("⚠️ 週報作成に失敗しました")
    else:
        print("📅 週報作成は不要です")

    print(f"[DONE] 全処理完了")


if __name__ == "__main__":
    main()
