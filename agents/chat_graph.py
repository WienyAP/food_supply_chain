
from typing import TypedDict, Callable
from langgraph.graph import StateGraph, START, END


# =========================================================
# STATE CHATBOT
# =========================================================

Handler = Callable[[str], str]


class ChatState(TypedDict, total=False):
    question: str
    intent: str
    answer: str
    handlers: dict[str, Handler]


# =========================================================
# 1. ROUTER
# =========================================================

def route_question(state: ChatState):
    """Menentukan intent pertanyaan berdasarkan kata kunci."""

    q = state.get("question", "").strip().lower()

    # Prioritas pertama: rekomendasi
    recommendation_words = [
        "rekomendasi",
        "pemerataan",
        "wilayah surplus",
        "wilayah defisit",
        "kandidat pasokan",
        "kandidat distribusi",
        "redistribusi",
    ]

    # Prioritas kedua: distribusi aktual
    distribution_words = [
        "distribusi",
        "rute",
        "pengiriman",
        "volume distribusi",
        "lokasi asal",
        "lokasi tujuan",
        "asal pasokan",
        "tujuan pengiriman",
        "catatan rute",
    ]

    # Prioritas ketiga: produksi
    supply_words = [
        "produksi",
        "peringkat",
        "tertinggi",
        "terendah",
        "tren",
        "trend",
        "bandingkan",
        "perbandingan",
        "selisih",
        "persentase",
        "kabupaten apa saja",
        "kota apa saja",
    ]

    if any(word in q for word in recommendation_words):
        intent = "recommendation"

    elif any(word in q for word in distribution_words):
        intent = "distribution"

    elif any(word in q for word in supply_words):
        intent = "supply"

    else:
        intent = "general"

    return {"intent": intent}


# =========================================================
# 2. EKSEKUSI HANDLER
# =========================================================

def run_handler(state: ChatState, intent: str):
    """Menjalankan handler sesuai intent yang dipilih router."""

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

    try:
        answer = handler(question)

        if answer is None or not str(answer).strip():
            answer = "Belum ada hasil yang dapat ditampilkan."

        return {"answer": str(answer)}

    except Exception as exc:
        return {
            "answer": (
                f"Terjadi kesalahan saat menjalankan handler "
                f"'{intent}': {exc}"
            )
        }


# =========================================================
# 3. NODE SETIAP AGENT
# =========================================================

def supply_node(state: ChatState):
    return run_handler(state, "supply")


def distribution_node(state: ChatState):
    return run_handler(state, "distribution")


def recommendation_node(state: ChatState):
    return run_handler(state, "recommendation")


def general_node(state: ChatState):
    return run_handler(state, "general")


# =========================================================
# 4. MEMBANGUN LANGGRAPH
# =========================================================

def build_chat_graph():
    graph = StateGraph(ChatState)

    graph.add_node("router", route_question)
    graph.add_node("supply", supply_node)
    graph.add_node("distribution", distribution_node)
    graph.add_node("recommendation", recommendation_node)
    graph.add_node("general", general_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["intent"],
        {
            "supply": "supply",
            "distribution": "distribution",
            "recommendation": "recommendation",
            "general": "general",
        },
    )

    graph.add_edge("supply", END)
    graph.add_edge("distribution", END)
    graph.add_edge("recommendation", END)
    graph.add_edge("general", END)

    return graph.compile()
