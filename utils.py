"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import time
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    
    return icon


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def get_llm_response(chat_message, progress_callback=None):
    """
    LLMからの回答取得

    Args:
        chat_message: ユーザー入力値
        progress_callback: 進捗表示用のコールバック関数

    Returns:
        LLMからの回答
    """
    start_time = time.time()
    
    # 進捗更新用のヘルパー関数
    def update_progress(message):
        if progress_callback:
            elapsed = time.time() - start_time
            progress_callback(f"{message} ({elapsed:.1f}秒経過)")
    
    # ステップ1: LLMオブジェクトの準備
    update_progress("🔧 AI モデルを準備しています...")
    llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)
    time.sleep(0.5)  # 視覚的な進捗確認のための短い待機

    # ステップ2: プロンプトテンプレートの作成
    update_progress("📝 プロンプトテンプレートを作成しています...")
    
    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # モードによってLLMから回答を取得する用のプロンプトを変更
    if st.session_state.mode == ct.ANSWER_MODE_1:
        # モードが「社内文書検索」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        # モードが「社内問い合わせ」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    # LLMから回答を取得する用のプロンプトテンプレートを作成
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    time.sleep(0.5)  # 視覚的な進捗確認のための短い待機

    # ステップ3: 関連文書の検索
    update_progress("🔍 関連文書を検索しています...")
    
    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )

    # ステップ4: 回答生成チェーンの作成
    update_progress("⚙️ 回答生成チェーンを構築しています...")
    
    # LLMから回答を取得する用のChainを作成
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    time.sleep(0.5)  # 視覚的な進捗確認のための短い待機

    # ステップ5: AI回答の生成
    update_progress("🤖 AI が回答を生成しています...")
    
    # LLMへのリクエストとレスポンス取得
    llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
    
    # ステップ6: 会話履歴の更新
    update_progress("💾 会話履歴を更新しています...")
    
    # LLMレスポンスを会話履歴に追加
    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])
    
    # 完了メッセージ
    total_time = time.time() - start_time
    update_progress(f"✅ 処理が完了しました！ (合計: {total_time:.1f}秒)")
    time.sleep(1)  # 完了メッセージを表示するための待機

    return llm_response