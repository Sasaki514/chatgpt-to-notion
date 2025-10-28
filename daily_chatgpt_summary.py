import os
import json
import requests
import argparse
import textwrap
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from openai import OpenAI

JST = timezone(timedelta(hours=9))


def jst_today():
    """JSTの今日の日付をYYYY-MM-DD形式で返す"""
    return datetime.now(JST).date().isoformat()


SYSTEM_PROMPT = """あなたは知識の振り返りを支援する専門家です。
以下のルールを厳守してください：

1. 生のログをそのまま出力してはいけません
2. 必ずMarkdown形式で出力してください
3. 各日付ごとに## 日付（YYYY-MM-DD）を必ず出力してください
4. 各日の相談内容を3-5個のトピックに整理してください
5. 各トピックの分野・カテゴリを特定してください
6. 要点は「新しく知った知識・ノウハウ・技術」に焦点を当ててください（用語の定義や内容説明も含める）
7. 次のアクションは「より深い学習・高度な調査」を提案してください
8. 会話時間を記録してください（複数会話の場合は時間帯の範囲）
9. 指定された形式以外は一切出力しないでください

【重要】出力は必ずMarkdown形式で行ってください。見出しには#記号を使用し、太字には**を使用してください。

出力形式（Markdown形式）：
## 日付（YYYY-MM-DD）

**〔相談したトピック名〕**
**分野:** ビジネス分野・技術分野・カテゴリ（例：プログラミング、データ分析、システム設計、AI/機械学習、Web開発、データベース、セキュリティ、インフラ、UI/UX、ビジネス分析など）
**時間:** 会話時間（例：14:30-15:45、複数会話の場合は 09:00-11:30, 14:00-16:00 など）
**新しく知った知識:** この相談で新たに得られた具体的な知識・ノウハウ・技術・手法（用語：説明、補足の形式で記載。例：「Selenium WebDriver：Webブラウザを自動制御するツール、今回は要素の取得方法を扱った」「正規表現の〜パターン：〜を意味する記法、今回は〜の使い方を扱った」など）
**次のアクション:** 具体的な深堀り学習提案（例：「この技術の他の活用場面を調べてみたら？-活用場面：具体的な業務やプロジェクトでの応用例」「より効率的な実装パターンを探してみたら？-実装パターン：コードの構造や設計手法」「関連する最新のベストプラクティスを調べてみたら？-ベストプラクティス：業界標準や推奨される手法」「この技術のパフォーマンス最適化について調べてみたら？-パフォーマンス最適化：処理速度やリソース効率の改善手法」など）

**〔相談したトピック名〕**
**分野:** ビジネス分野・技術分野・カテゴリ
**時間:** 会話時間
**新しく知った知識:** この相談で新たに得られた具体的な知識・ノウハウ・技術・手法（用語：説明、補足の形式で記載）
**次のアクション:** 具体的な深堀り学習提案（「〜について調べてみたら？」の形式で提案し、-用語：説明で提案内容の詳細を記載）

（以下、他の日付についても同様のMarkdown形式で続ける）"""

USER_PROMPT_TEMPLATE = "以下の相談ログを要約してください：\n\n{raw_text}"

# 週報用プロンプト
WEEKLY_SYSTEM_PROMPT = """あなたは週間学習レポートを作成する専門家です。
以下のルールを厳守してください：

1. 生のログをそのまま出力してはいけません
2. 必ずMarkdown形式で出力してください
3. 週間の学習内容を振り返り、成果と課題を整理してください
4. 指定された形式以外は一切出力しないでください

【重要】出力は必ずMarkdown形式で行ってください。見出しには#記号を使用し、太字には**を使用してください。

出力形式（Markdown形式）：
## 週間学習レポート（YYYY年MM月第X週）

### 📊 学習サマリー
**学習日数:** X日間
**総学習時間:** 約X時間
**主要分野:** 分野1, 分野2, 分野3

### 🎯 今週の主要成果
- **技術的成果:** 習得した技術・解決した問題
- **知識的成果:** 新たに理解した概念・理論
- **実践的成果:** 実際に作成・改善したもの

### 📈 学習パターン分析
**集中時間帯:** 最も集中できた時間帯
**効率的な学習方法:** 効果的だった学習アプローチ
**学習の質:** 深い理解が得られた分野・トピック

### 🔍 今週の課題・つまずきポイント
- **技術的課題:** 解決できなかった問題・理解が浅い部分
- **学習方法の課題:** 効率が悪かった学習方法
- **知識のギャップ:** 不足している基礎知識

### 🚀 来週の学習計画
**重点学習分野:** 来週重点的に取り組む分野
**具体的な学習目標:** 達成したい具体的な目標
**学習方法の改善:** より効率的な学習方法の試行

### 💡 今週の学びのハイライト
**最も印象的だった学習内容:** 今週最も価値があった学習
**新たな発見:** 新しい視点・気づき
**次への展望:** 今後の学習への期待・目標"""

WEEKLY_USER_PROMPT_TEMPLATE = "以下の週間相談ログを分析して週間学習レポートを作成してください：\n\n{raw_text}"


# 週報作成関数
def create_weekly_report(raw_text, api_key, model):
    """週間学習レポートを作成"""
    if not api_key or not OpenAI:
        return None

    client = OpenAI(api_key=api_key)
    user_prompt = WEEKLY_USER_PROMPT_TEMPLATE.format(raw_text=raw_text)

    print(f"📤 週報作成のためChatGPT API投げました:")
    print(f"   モデル: {model}")
    print(f"   入力テキスト長: {len(raw_text)}文字")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": WEEKLY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    result = resp.choices[0].message.content.strip()
    print(f"📥 週報作成完了: {len(result)}文字")
    return result

# 最終週報登録日管理


def get_last_weekly_report_date(workdir):
    """最終週報登録日を取得"""
    config_file = os.path.join(workdir, "weekly_report_config.json")
    old_txt_file = os.path.join(workdir, "last_weekly_report.txt")

    # 既存のtxtファイルがある場合は移行
    if os.path.exists(old_txt_file) and not os.path.exists(config_file):
        try:
            with open(old_txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    # JSONファイルを作成
                    config = {
                        "last_weekly_report_date": content,
                        "migrated_at": datetime.now(JST).isoformat(),
                        "version": "1.0"
                    }
                    with open(config_file, "w", encoding="utf-8") as json_f:
                        json.dump(config, json_f, ensure_ascii=False, indent=2)
                    print(
                        f"✅ 設定ファイルをJSON形式に移行しました: {old_txt_file} → {config_file}")
                    return content
        except Exception as e:
            print(f"⚠️ 設定ファイル移行に失敗: {e}")

    # JSONファイルから読み取り
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get("last_weekly_report_date")
        except (json.JSONDecodeError, KeyError):
            return None
    return None


def save_last_weekly_report_date(workdir, date_str):
    """最終週報登録日を保存"""
    config_file = os.path.join(workdir, "weekly_report_config.json")

    # 既存の設定を読み込み（存在しない場合は新規作成）
    config = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, KeyError):
            config = {}

    # 設定を更新
    config.update({
        "last_weekly_report_date": date_str,
        "updated_at": datetime.now(JST).isoformat(),
        "version": "1.0"
    })

    # 設定を保存
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def should_create_weekly_report(workdir):
    """週報作成が必要かチェック"""
    today = jst_today()
    today_date = datetime.strptime(today, "%Y-%m-%d").date()

    # 最終登録日をチェック
    last_weekly_date = get_last_weekly_report_date(workdir)

    if last_weekly_date:
        # 1. last_weekly_report.txtの日付がある場合
        last_date = datetime.strptime(last_weekly_date, "%Y-%m-%d").date()

        # その日付の次の土曜日を取得
        last_date_weekday = last_date.weekday()
        next_saturday = last_date - \
            timedelta(days=last_date_weekday) + timedelta(days=5)
        if next_saturday <= last_date:
            next_saturday += timedelta(days=7)

        # システム日付が次の土曜以降であれば週報登録
        if today_date >= next_saturday:
            return True
        else:
            # システム日付が次の土曜以前であれば週報登録なし
            return False
    else:
        # 2. last_weekly_report.txtの日付がない場合
        # システム日付の前の金曜までの週報を登録
        return has_sufficient_weekly_data(workdir)


def get_weekly_date_range():
    """今週の日付範囲を取得（月曜日〜金曜日）"""
    today = jst_today()
    today_date = datetime.strptime(today, "%Y-%m-%d").date()

    # 今週の月曜日を取得
    days_since_monday = today_date.weekday()
    monday = today_date - timedelta(days=days_since_monday)

    # 今週の金曜日を取得
    friday = monday + timedelta(days=4)

    return monday.strftime("%Y-%m-%d"), friday.strftime("%Y-%m-%d")


def get_latest_friday_date():
    """直近の金曜日の日付を取得"""
    today = jst_today()
    today_date = datetime.strptime(today, "%Y-%m-%d").date()

    # 今週の金曜日を取得
    days_since_monday = today_date.weekday()
    monday = today_date - timedelta(days=days_since_monday)
    friday = monday + timedelta(days=4)

    # 今日が金曜日より前の場合は、前週の金曜日を取得
    if today_date < friday:
        friday -= timedelta(days=7)

    return friday.strftime("%Y-%m-%d")


def has_sufficient_weekly_data(workdir):
    """週報作成に十分なデータがあるかチェック"""
    # システム日付の前の金曜日までの会話データがあるかチェック
    latest_friday = get_latest_friday_date()
    print(f"   前の金曜日: {latest_friday}")

    # 実際のデータチェックは、ChatGPTログの読み取り時に実装
    # ここでは常にTrueを返す（実際のデータは後でチェック）
    return True


def get_db_props(notion_token: str, database_id: str):
    h = {"Authorization": f"Bearer {notion_token}",
         "Notion-Version": "2022-06-28"}
    r = requests.get(
        f"https://api.notion.com/v1/databases/{database_id}", headers=h)
    if r.status_code != 200:
        raise RuntimeError(f"DBメタ取得に失敗: {r.status_code} {r.text}")
    data = r.json()
    title_prop = None
    date_prop = None
    for name, prop in data.get("properties", {}).items():
        if prop.get("type") == "title" and not title_prop:
            title_prop = name
        if prop.get("type") == "date" and not date_prop:
            date_prop = name
    if not title_prop:
        raise RuntimeError("タイトル型プロパティが見つかりません（Notion側でTitle列が必要）")
    return title_prop, date_prop


def create_page(notion_token: str, database_id: str, title_prop: str, date_prop: str, title_text: str, date_iso: str, md_body: str):
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }

    properties = {
        title_prop: {"title": [{"text": {"content": title_text}}]}
    }
    if date_prop:
        # date_isoが辞書の場合はそのまま使用、文字列の場合はstartとして設定
        if isinstance(date_iso, dict):
            properties[date_prop] = {"date": date_iso}
        else:
            properties[date_prop] = {"date": {"start": date_iso}}

    # Markdownはそのままブロックとして貼る（code: markdown で可読性担保）
    children = [{
        "object": "block",
        "type": "code",
        "code": {
            "language": "markdown",
            "rich_text": [{"type": "text", "text": {"content": md_body[:200000]}}]
        }
    }]

    payload = {"parent": {"database_id": database_id},
               "properties": properties,
               "children": children}

    r = requests.post("https://api.notion.com/v1/pages", headers=headers,
                      data=json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    if r.status_code not in (200, 201):
        raise RuntimeError(f"ページ作成に失敗: {r.status_code} {r.text}")
    return r.json()


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="ChatGPT当日相談内容まとめ→Notionに保存")
    parser.add_argument("--date", "-d", default=None,
                        help="保存日(YYYY-MM-DD)。未指定はJSTの今日。")
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY が未設定です（.env）")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # コスト重視の既定

    notion_token = os.getenv("NOTION_TOKEN")
    database_id = os.getenv("DATABASE_ID")
    if not notion_token or not database_id:
        raise RuntimeError("NOTION_TOKEN / DATABASE_ID が未設定です（.env）")

    # 1. ChatGPT APIで当日の相談内容をまとめる
    print("ChatGPT APIで当日の相談内容をまとめています...")

    # 実際の相談ログを取得（ここではサンプルテキストを使用）
    # 実際の実装では、ChatGPTの会話履歴を取得する処理が必要
    sample_raw_text = """
    [ユーザー] formCompTra.pyの冗長性を解消したいです。共通処理をまとめられませんか？
    [アシスタント] はい、段階的クリック処理などの共通部分を別ファイルに関数化しましょう。common_actions.pyを作成して、click_show_all_button、click_compliance_training_linkなどの関数を実装します。
    
    [ユーザー] ChromeDriverManagerが起動しない問題があります。
    [アシスタント] 診断スクリプトを作成して問題を特定しましょう。chrome_diagnostic.pyでシステム情報、Chrome設定、ドライバー起動テストを行います。
    
    [ユーザー] ChatGPTのAPIログが残らないと聞きました。
    [アシスタント] client.responses.create()ではなく、client.chat.completions.create()を使用する必要があります。前者はログに残らない可能性があります。
    
    [ユーザー] デバッグ出力が冗長すぎて見にくいです。
    [アシスタント] シンプルなログ形式に変更しましょう。📤 API投げる → 📥 レスポンス → 📋 要約結果 → 📝 Notion登録の流れで整理します。
    """

    # ChatGPTに投げる（Chat Completions API）
    client = OpenAI(api_key=openai_key)
    user_prompt = USER_PROMPT_TEMPLATE.format(raw_text=sample_raw_text)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3  # より一貫した出力のため
    )
    recap_markdown = resp.choices[0].message.content.strip()

    # デバッグ用: ChatGPTの出力を確認
    print("=" * 50)
    print("ChatGPTの出力:")
    print(recap_markdown)
    print("=" * 50)

    # 2. Notion APIでデータベースに記録
    print("Notionデータベースに記録しています...")
    date_iso = (datetime.now(JST).date().isoformat()
                if not args.date else args.date)
    title_text = f"{date_iso} ChatGPT相談まとめ"

    # 23:00〜23:30の固定時間を設定（JST）
    date_obj = datetime.strptime(date_iso, "%Y-%m-%d").date()
    start_time = datetime.combine(
        date_obj, datetime.min.time().replace(hour=23, minute=0)).replace(tzinfo=JST)
    end_time = datetime.combine(
        date_obj, datetime.min.time().replace(hour=23, minute=30)).replace(tzinfo=JST)

    date_with_time = {
        "start": start_time.isoformat(),
        "end": end_time.isoformat()
    }

    title_prop, date_prop = get_db_props(notion_token, database_id)
    page = create_page(notion_token, database_id, title_prop,
                       date_prop, title_text, date_with_time, recap_markdown)

    print("✅ 保存完了")
    print("ページURL:", page.get("url"))

    # 3. 週報作成チェック
    print("\n=== 週報作成チェック ===")
    workdir = os.getenv("WORK_DIR", os.path.join(
        os.path.dirname(__file__), "ChatGPT_Notion"))

    # 週報作成判断の詳細ログ
    print("🔍 週報作成判断プロセス:")
    last_weekly_date = get_last_weekly_report_date(workdir)
    print(f"   最終登録日: {last_weekly_date if last_weekly_date else '未登録'}")

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

        # 週間の相談ログを取得（実際の実装では、指定期間のログを取得）
        # ここではサンプルとして、複数日のログを模擬
        weekly_raw_text = f"""
        === 週間相談ログ ({monday} 〜 {friday}) ===
        
        {monday}:
        [ユーザー] ChromeDriverManagerが起動しない問題があります。
        [アシスタント] 診断スクリプトを作成して問題を特定しましょう。
        
        {friday}:
        [ユーザー] 週報機能を追加したいです。
        [アシスタント] 週間学習レポートのプロンプトと管理機能を実装しましょう。
        """

        # 週報作成
        weekly_report = create_weekly_report(
            weekly_raw_text, openai_key, model)

        if weekly_report:
            # 週報をNotionに登録
            today = jst_today()
            weekly_title = f"{today} 週間学習レポート"

            # 土曜日の12:00〜13:00の時間設定
            date_obj = datetime.strptime(today, "%Y-%m-%d").date()
            start_time = datetime.combine(
                date_obj, datetime.min.time().replace(hour=12, minute=0)).replace(tzinfo=JST)
            end_time = datetime.combine(
                date_obj, datetime.min.time().replace(hour=13, minute=0)).replace(tzinfo=JST)

            weekly_date_with_time = {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }

            weekly_page = create_page(notion_token, database_id, title_prop,
                                      date_prop, weekly_title, weekly_date_with_time, weekly_report)

            # 最終週報登録日を保存
            save_last_weekly_report_date(workdir, today)

            print("✅ 週報保存完了")
            print("週報ページURL:", weekly_page.get("url"))
        else:
            print("⚠️ 週報作成に失敗しました")
    else:
        print("📅 週報作成は不要です")


if __name__ == "__main__":
    main()
