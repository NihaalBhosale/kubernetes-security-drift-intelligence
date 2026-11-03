"""Streamlit dashboard for Kubernetes Security Drift Intelligence."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from app.drift_detector import DriftDetectionEngine
from app.rule_engine import SecurityRuleEngine
from app.attack_path_analyzer import AttackPathAnalyzer
from app.risk_assesment import calculate_risk

st.set_page_config(page_title="Kubernetes Security Drift Intelligence", page_icon="🛡️", layout="wide")
st.title("Kubernetes Security Drift Intelligence")
st.caption("Live Kubernetes configuration drift · Rule-based risk assessment · Attack-path analysis")

baseline_path = ROOT / "snapshots" / "baseline.json"
current_path = ROOT / "snapshots" / "current.json"
alerts_path = ROOT / "data" / "alerts.json"

def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default

baseline = read_json(baseline_path, {"resources": []})
current = read_json(current_path, {"resources": []})
alerts = read_json(alerts_path, [])
events = DriftDetectionEngine().detect_drift(baseline, current)
rules_engine = SecurityRuleEngine(ROOT / "rules" / "security_rules.yaml")
analyzer = AttackPathAnalyzer()
graph = analyzer.build_graph(current)
paths = analyzer.find_attack_paths()

matched_rows = []
for event in events:
    for rule in rules_engine.evaluate_event(event):
        risk = calculate_risk(rule["impact"], bool(paths))
        matched_rows.append({
            "Risk": risk["level"], "Score": risk["score"],
            "Rule": rule["rule_id"], "Resource": rule["resource"],
            "Field": event.field, "Old value": str(event.old_value),
            "New value": str(event.new_value), "Drift ID": event.drift_id
        })

c1, c2, c3, c4 = st.columns(4)
c1.metric("Baseline resources", len(baseline.get("resources", [])))
c2.metric("Current resources", len(current.get("resources", [])))
c3.metric("Drift events", len(events))
c4.metric("Open alerts", sum(1 for a in alerts if a.get("status") == "Open"))

tab1, tab2, tab3 = st.tabs(["Security Overview", "Attack Graph", "Alerts"])

with tab1:
    st.subheader("Detected configuration drift")
    if events:
        drift_df = pd.DataFrame([{
            "Kind": e.resource_kind, "Namespace": e.namespace, "Resource": e.resource_name,
            "Field": e.field, "Change": e.change_type, "Old value": str(e.old_value),
            "New value": str(e.new_value), "Severity": e.severity
        } for e in events])
        st.dataframe(drift_df, use_container_width=True, hide_index=True)
    else:
        st.success("No drift detected between the baseline and current snapshot.")

    st.subheader("Security rule matches")
    if matched_rows:
        st.dataframe(pd.DataFrame(matched_rows), use_container_width=True, hide_index=True)
        risk_counts = pd.Series([r["Risk"] for r in matched_rows]).value_counts()
        st.bar_chart(risk_counts)
    else:
        st.info("No configured security rules matched the detected drift.")

    with st.expander("Snapshot resource inventory"):
        st.write("Baseline")
        st.json(baseline.get("resources", []))
        st.write("Current")
        st.json(current.get("resources", []))

with tab2:
    st.subheader("Attack-path candidate graph")
    st.caption("Simplified view: only connected resources are shown. Relationships are heuristic, not proof of exploitability.")

    # Hide isolated cluster-wide objects (often dozens of unused built-in RBAC objects).
    connected_nodes = [node for node, degree in graph.degree() if degree > 0]
    visible_graph = graph.subgraph(connected_nodes).copy()

    # Prefer paths that actually connect an external entry point to a sensitive target.
    # If none exist, show only the small connected portion of the graph rather than every resource.
    if paths:
        path_nodes = {node for path in paths[:5] for node in path}
        candidate_graph = graph.subgraph(path_nodes).copy()
        if candidate_graph.number_of_nodes() > 0:
            visible_graph = candidate_graph

    if visible_graph.number_of_nodes() == 0:
        st.info("No connected attack relationships were found. This is different from proving the cluster is secure.")
    else:
        # Short labels keep Kubernetes namespace and resource names readable.
        def short_label(node):
            if node == "EXTERNAL":
                return "External\nEntry"
            if node.startswith("SENSITIVE:"):
                return node.replace("SENSITIVE: ", "Risk: ").replace("elevated-container", "Elevated\ncontainer")
            parts = node.split("/")
            if len(parts) >= 3:
                kind, namespace, name = parts[0], parts[1], parts[2]
                return f"{kind}\n{name}\n(ns: {namespace})"
            return node

        # Use a layered layout when a directed acyclic view is available; otherwise use a
        # deterministic spring layout. This is more stable and easier to read than plotting
        # every node in the complete cluster graph.
        try:
            if nx.is_directed_acyclic_graph(visible_graph):
                pos = nx.multipartite_layout(visible_graph, subset_key=lambda n: nx.shortest_path_length(visible_graph, "EXTERNAL", n) if "EXTERNAL" in visible_graph and nx.has_path(visible_graph, "EXTERNAL", n) else 0, align="horizontal", scale=2.0)
            else:
                pos = nx.spring_layout(visible_graph, seed=11, k=1.8, iterations=100)
        except (nx.NetworkXError, nx.NodeNotFound):
            pos = nx.spring_layout(visible_graph, seed=11, k=1.8, iterations=100)

        fig, ax = plt.subplots(figsize=(12, 6.5))
        colors = []
        sizes = []
        for node in visible_graph.nodes:
            kind = visible_graph.nodes[node].get("kind", "")
            if node == "EXTERNAL":
                colors.append("#f4a261"); sizes.append(2600)
            elif kind == "SensitiveTarget":
                colors.append("#e76f51"); sizes.append(2800)
            elif kind == "Pod":
                colors.append("#70b7d6"); sizes.append(2300)
            elif kind in ("Service", "Ingress"):
                colors.append("#8ecae6"); sizes.append(2200)
            elif kind in ("Role", "ClusterRole", "RoleBinding", "ClusterRoleBinding"):
                colors.append("#c7d9b4"); sizes.append(1800)
            else:
                colors.append("#b9c6d8"); sizes.append(1800)

        nx.draw_networkx_nodes(visible_graph, pos, ax=ax, node_color=colors, node_size=sizes,
                               edgecolors="#52606d", linewidths=1.0)
        nx.draw_networkx_labels(visible_graph, pos, ax=ax,
                                labels={node: short_label(node) for node in visible_graph.nodes},
                                font_size=8, font_weight="medium")
        nx.draw_networkx_edges(visible_graph, pos, ax=ax, arrows=True, arrowsize=18,
                               edge_color="#7b8794", width=1.5, connectionstyle="arc3,rad=0.04",
                               min_source_margin=20, min_target_margin=20)
        edge_labels = {(u, v): data.get("relation", "related").replace("-", " ")
                       for u, v, data in visible_graph.edges(data=True)}
        nx.draw_networkx_edge_labels(visible_graph, pos, edge_labels=edge_labels, ax=ax,
                                     font_size=7, rotate=False, label_pos=0.5,
                                     bbox={"alpha": 0.8, "color": "white", "pad": 1})
        ax.margins(0.18)
        ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.write(f"Showing **{visible_graph.number_of_nodes()} connected nodes** and **{visible_graph.number_of_edges()} relationships** (filtered from {graph.number_of_nodes()} total resource nodes).")

    if paths:
        st.subheader("Candidate paths to sensitive targets")
        for i, path in enumerate(paths[:5], start=1):
            st.markdown(f"**Path {i}**")
            st.code(" → ".join(short_label(node).replace("\n", " ") for node in path), language="text")
    else:
        st.info("No external-to-sensitive-target path was identified by the current heuristic. The simplified graph shows only connected relationships.")

with tab3:
    st.subheader("Stored alerts")
    if alerts:
        st.dataframe(pd.DataFrame(alerts), use_container_width=True, hide_index=True)
    else:
        st.info("No alerts are stored yet. Run the analysis pipeline to populate alerts.")
    st.caption(f"Alert store: {alerts_path.relative_to(ROOT)}")

st.sidebar.header("Run instructions")
st.sidebar.code("python -m app.main --live --namespace default", language="powershell")
st.sidebar.write("Run the command in a separate terminal to refresh `snapshots/current.json`, then refresh this dashboard.")
st.sidebar.warning("This dashboard reads the saved snapshots and alert file. It does not itself trigger Kubernetes collection.")


# ----------------------------------------------------------------------------------------- dynamic engine
st.divider()
st.header("Dynamic engine (proposed)")
bundle = ROOT / "models" / "dynamic_bundle.pkl"
if not bundle.exists():
    st.info("Train it first: `python -m experiments.run_all` (creates models/dynamic_bundle.pkl).")
elif not baseline.get("resources") or not current.get("resources"):
    st.info("Need both snapshots.")
else:
    try:
        from app.dynamic.engine import DynamicRiskEngine
        res = DynamicRiskEngine.load(bundle).assess(baseline, current)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Alert", "YES" if res["alert"] else "no")
        d2.metric("Probability dangerous", f"{res['probability']:.2f}")
        d3.metric("Est. damage increase", f"{res['estimated_damage_increase']:.2f} node-eq")
        d4.metric("Severity", res["severity"])
        st.caption(f"{res['active_rules']} active rules for this cluster; key parameters: " + ", ".join(res["active_parameters"][:8]))
        if res["rules_fired"]:
            st.dataframe(pd.DataFrame([{"Rule": f["rule"], "Tactic": f["tactic"], "Dynamic weight": round(f["weight_introduced"], 2),
                                        "Contribution": round(f["contribution"], 2), "New instances": ", ".join(map(str, f["introduced"])),
                                        "Description": f["description"]} for f in res["rules_fired"]]), use_container_width=True)
        else:
            st.success("No rule evidence in this drift.")
    except Exception as exc:  # keep the original dashboard usable
        st.warning(f"Dynamic engine unavailable: {exc}")
