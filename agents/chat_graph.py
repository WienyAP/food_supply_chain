from typing import TypedDict, Callable
from langgraph.graph import StateGraph, START, END


Handler = Callable[[str], str]


class ChatState(TypedDict, total=False):
    question: str
    intent: str
    answer: str
    handlers: dict[str, Handler]
    previous_question: str
    previous_intent: str


# =========================================================
# 1. ROUTER
# =========================================================

def route_question(state: ChatState):
    q = state.get("question", "").strip().lower()

    month_words = list([
        "januari", "februari", "maret", "april", "mei", "juni",
        "juli", "agustus", "september", "oktober", "november", "desember",
        "jan", "feb", "mar", "apr", "jun", "jul", "agu", "sep", "okt", "nov", "des"
    ])
    import re
    has_month = any(re.search(rf"\b{re.escape(word)}\b", q) for word in month_words)

    balance_words = [
        "balance", "neraca", "surplus", "defisit", "kekurangan",
        "kelebihan produksi", "produksi dan kebutuhan", "produksi dikurangi kebutuhan"
    ]
    demand_words = ["kebutuhan", "permintaan", "konsumsi", "demand"]
    monthly_supply_words = ["produksi", "panen", "produksi beras"]

    recommendation_words = [
        "rekomendasi", "pemerataan", "wilayah surplus",
        "wilayah defisit", "kandidat pasokan",
        "kandidat distribusi", "redistribusi",
    ]

    distribution_words = [
        "distribusi", "rute", "pengiriman",
        "volume distribusi", "lokasi asal",
        "lokasi tujuan", "asal pasokan",
        "tujuan pengiriman", "catatan rute",
    ]

    supply_words = [
        "produksi", "peringkat", "tertinggi", "terendah",
        "tren", "trend", "bandingkan", "perbandingan",
        "selisih", "persentase", "pasokan", "surplus",
        "defisit", "kebutuhan", "konsumsi",
        "kabupaten apa saja", "kota apa saja",
    ]

    # Prioritaskan intent yang lebih spesifik sebelum intent umum.
    if any(word in q for word in recommendation_words):
        intent = "recommendation"

    elif any(word in q for word in distribution_words):
        intent = "distribution"

    elif any(word in q for word in balance_words):
        intent = "balance"

    elif any(word in q for word in demand_words) and (has_month or any(y in q for y in ["2021", "2022", "2023", "2024", "2025"])):
        intent = "monthly_demand"

    elif has_month and any(word in q for word in monthly_supply_words):
        intent = "monthly_supply"

    elif any(word in q for word in supply_words):
        intent = "supply"

    else:
        # Pertanyaan lanjutan pendek mengikuti topik sebelumnya
        previous_intent = state.get("previous_intent", "general")

        follow_up_words = [
            "bandingkan", "bagaimana dengan", "kalau",
            "lalu", "mana yang", "lebih tinggi",
            "lebih rendah", "selisihnya", "daerah itu",
            "wilayah tersebut", "tahun sebelumnya",
        ]

        if any(word in q for word in follow_up_words):
            intent = previous_intent
        else:
            intent = "general"

    return {"intent": intent}


# =========================================================
# 2. EKSEKUSI HANDLER
# =========================================================

def run_handler(state: ChatState, intent: str):
    handlers = state.get("handlers", {})
    handler = handlers.get(intent)

    if handler is None:
        return {
            "answer": (
                f"Handler untuk intent '{intent}' belum dihubungkan. "
                "Periksa konfigurasi handler di app.py."
            )
        }

    question = state.get("question", "").strip()

    # Tambahkan konteks sebelumnya untuk pertanyaan lanjutan.
    previous_question = state.get("previous_question", "").strip()
    follow_up_starters = (
        "bandingkan", "bagaimana dengan", "kalau",
        "lalu", "mana yang", "lebih tinggi",
        "lebih rendah", "selisihnya", "tahun sebelumnya",
    )

    if previous_question and question.lower().startswith(follow_up_starters):
        question = (
            f"Konteks pertanyaan sebelumnya: {previous_question}\n"
            f"Pertanyaan lanjutan: {question}"
        )

    try:
        answer = handler(question)

        if answer is None or not str(answer).strip():
            answer = "Belum ada hasil yang dapat ditampilkan."

        return {"answer": str(answer)}

    except Exception as exc:
        return {"answer": f"Terjadi kesalahan pada handler '{intent}': {exc}"}


def supply_node(state: ChatState):
    return run_handler(state, "supply")


def monthly_supply_node(state: ChatState):
    return run_handler(state, "monthly_supply")


def monthly_demand_node(state: ChatState):
    return run_handler(state, "monthly_demand")


def balance_node(state: ChatState):
    return run_handler(state, "balance")


def distribution_node(state: ChatState):
    return run_handler(state, "distribution")


def recommendation_node(state: ChatState):
    return run_handler(state, "recommendation")


def general_node(state: ChatState):
    return run_handler(state, "general")


# =========================================================
# 3. MEMBANGUN LANGGRAPH
# =========================================================

def build_chat_graph():
    graph = StateGraph(ChatState)

    graph.add_node("router", route_question)
    graph.add_node("supply", supply_node)
    graph.add_node("monthly_supply", monthly_supply_node)
    graph.add_node("monthly_demand", monthly_demand_node)
    graph.add_node("balance", balance_node)
    graph.add_node("distribution", distribution_node)
    graph.add_node("recommendation", recommendation_node)
    graph.add_node("general", general_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["intent"],
        {
            "supply": "supply",
            "monthly_supply": "monthly_supply",
            "monthly_demand": "monthly_demand",
            "balance": "balance",
            "distribution": "distribution",
            "recommendation": "recommendation",
            "general": "general",
        },
    )

    for node in [
        "supply", "monthly_supply", "monthly_demand", "balance",
        "distribution", "recommendation", "general"
    ]:
        graph.add_edge(node, END)

    return graph.compile()
