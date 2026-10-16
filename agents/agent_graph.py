from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

from agents.supply_agent import SupplyAgent
from agents.demand_agent import DemandAgent
from agents.distribution_agent import DistributionAgent
from agents.coordinator_agent import CoordinatorAgent
from agents.recommendation_agent import RecommendationAgent


class AgriFoodState(TypedDict, total=False):
    production: pd.DataFrame
    consumption: pd.DataFrame
    population: pd.DataFrame
    distribution_raw: pd.DataFrame

    consumption_unit: str | None

    supply_result: pd.DataFrame
    demand_result: pd.DataFrame | None
    distribution_result: pd.DataFrame

    coordinator_result: dict[str, Any]
    recommendation_result: pd.DataFrame

    llm_report: str
    errors: list[str]


def build_agri_food_graph():
    supply_agent = SupplyAgent()
    demand_agent = DemandAgent()
    distribution_agent = DistributionAgent()
    coordinator_agent = CoordinatorAgent()
    recommendation_agent = RecommendationAgent()


    def supply_node(state: AgriFoodState):
        result = supply_agent.run(state['production'])
        return {"supply_result": result}

    def demand_node(state: AgriFoodState):
        unit = state.get("consumption_unit")

        # Jangan menghitung kebutuhan jika unit belum terverifikasi.
        if unit != "kg_per_capita_per_year":
            return {"demand_result": None}

        result = demand_agent.run({
            "consumption": state["consumption"],
            "population": state["population"],
            "consumption_unit": unit,
        })
        return {"demand_result": result}

    def distribution_node(state: AgriFoodState):
        result = distribution_agent.run(
            state["distribution_raw"]
        )
        return {"distribution_result":  result}

    def coordinator_node(state: AgriFoodState):
        result = coordinator_agent.run({
            "supply": state["supply_result"],
            "demand": state.get("demand_result"),
            "distribution": state["distribution_result"],
        })

        return {"coordinator_result": result}


    def recommendation_node(state: AgriFoodState):
        coordinator_result = state.get("coordinator_result", {})

        result = recommendation_agent.run({
            "supply": state["supply_result"],
            "demand": state.get("demand_result"),
            "distribution": state["distribution_result"],
        })

        return {"recommendation_result": result}


    def llm_node(state: AgriFoodState):
        supply = state["supply_result"]
        demand = state.get("demand_result")
        recommendations = state["recommendation_result"]

        supply_summary = (
            supply.groupby("year", as_index=False)
            .agg(
                jumlah_wilayah=("region_key", "nunique"),
                total_produksi=("produksi_beras", "sum"),
            )
            .tail(6)
            .to_dict(orient="records")
        )

        if demand is None:
            demand_summary = (
                "Estimasi konsumsi belum tersedia karena satuan"
                "Konsumsi_Pangan belum dikonfirmasi"
            )
        else:
            demand_summary = (
                demand.groupby("year", as_index=False)
                .agg(
                    estimasi_konsumsi_ton=(
                        "estimasi_konsumsi_ton_tahun", "sum"
                    )
                )
                .to_dict(orient="records")
            )

        recommendation_sumary = recommendations.head(10).to_dict(
            orient="records"
        )

        prompt = f"""
Anda adalah analisis rantai pasokan beras untuk wilayah Jawa Timur.

Jelaskan hasil berdasarkan data berikut saja.

Ringkasan produksi: {supply_summary}

Ringkasan konsumsi: {demand_summary}

Kandidat rekomendasi: {recommendation_sumary}

Ketentuan:
- Jangan mengarang angka, data, rute, atau fakta.
- Bedakan kandidat rekomendasi dari keputusan distribusi nyata
- Status produksi tinggi/rendah adalah status relatif historis, bukan bukti surplus atau defisit NBM.
- Jika estimasi konsumsi belum tersedia, nyatakan dengan jelas
- Data ini belum membentuk Neraca Bahan Makanan (NBM) lengkap
- Tulis dalam bahasa Indonesia dengan bagian:
1. Ringkasan hasil
2. Temuan utama
3. Kandidat rekomendasi
4. Keterbatasan data
"""

        try:
            llm = ChatOllama(
                model="llama3.2",
                temperature=0,
            )
            response = llm.invoke(prompt)
            report = response.content
        except Exception as exc:
            report = (
                "Laporan LLM belum tersedia"
                f"Periksa apakah Ollama berjalan dan model tersedia."
                f"Detail error: {exc}"
            )

        return {"llm_report": report}

    graph = StateGraph(AgriFoodState)

    graph.add_node("supply", supply_node)
    graph.add_node("demand", demand_node)
    graph.add_node("distribution", distribution_node)
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("recommendation", recommendation_node)
    graph.add_node("llm_report", llm_node)

    graph.add_edge(START, "supply")
    graph.add_edge("supply", "demand")
    graph.add_edge("demand", "distribution")
    graph.add_edge("distribution", "coordinator")
    graph.add_edge("coordinator", "recommendation")

    graph.add_edge("recommendation", "llm_report")
    graph.add_edge("llm_report", END)

    return graph.compile()
        
