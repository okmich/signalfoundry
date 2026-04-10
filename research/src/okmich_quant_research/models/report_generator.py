from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

from .experiment_runner import ExperimentResult


class ReportGenerator:
    """
    Generate HTML reports for experiments.

    Examples
    --------
    >>> generator = ReportGenerator()
    >>> generator.generate_report(result, output_path='report.html')
    """

    def __init__(self):
        """Initialize report generator."""
        pass

    def generate_report(self, result: ExperimentResult, output_path: Optional[str] = None, include_plots: bool = True) -> str:
        if output_path is None:
            output_path = Path(result.output_dir) / "report.html"
        else:
            output_path = Path(output_path)

        print(f"Generating report: {output_path}")
        # Generate HTML
        html = self._generate_html(result, include_plots)

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"   [OK] Report saved")
        return str(output_path)

    def _generate_html(self, result: ExperimentResult, include_plots: bool) -> str:
        """Generate HTML content."""
        best_model = result.get_best_model()

        # Prepare values
        best_model_name = best_model.model_name if best_model else "N/A"
        best_score = f"{best_model.composite_score:.4f}" if best_model else "N/A"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Report: {result.experiment_name}</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Experiment Report</h1>
            <h2>{result.experiment_name}</h2>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <section class="summary">
            <h2>Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <span class="label">Best Model</span>
                    <span class="value">{best_model_name}</span>
                </div>
                <div class="summary-item">
                    <span class="label">Composite Score</span>
                    <span class="value">{best_score}</span>
                </div>
                <div class="summary-item">
                    <span class="label">Models Trained</span>
                    <span class="value">{len(result.trained_models)}</span>
                </div>
                <div class="summary-item">
                    <span class="label">Features Selected</span>
                    <span class="value">{len(result.selected_features)}</span>
                </div>
            </div>
        </section>

        <section class="rankings">
            <h2>Model Rankings</h2>
            {self._generate_rankings_table(result.rankings)}
        </section>

        <section class="objectives">
            <h2>Best Model Objectives</h2>
            {self._generate_objectives_table(best_model) if best_model else '<p>No models available</p>'}
        </section>

        <section class="regime-stats">
            <h2>Regime Statistics</h2>
            {self._generate_regime_stats_table(result)}
        </section>

        <section class="features">
            <h2>Selected Features</h2>
            {self._generate_features_list(result.selected_features)}
        </section>

        <section class="config">
            <h2>Configuration</h2>
            {self._generate_config_section(result.config)}
        </section>

        <footer>
            <p>Generated with Claude Code Model Research Framework</p>
            <p>Output directory: {result.output_dir}</p>
        </footer>
    </div>
</body>
</html>"""

        return html

    def _get_css(self) -> str:
        """Get CSS styles."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        header {
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }

        h1 {
            color: #007acc;
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        h2 {
            color: #333;
            font-size: 1.8em;
            margin-bottom: 15px;
            margin-top: 30px;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }

        .timestamp {
            color: #666;
            font-size: 0.9em;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .summary-item {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007acc;
        }

        .summary-item .label {
            display: block;
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }

        .summary-item .value {
            display: block;
            font-size: 1.5em;
            font-weight: bold;
            color: #007acc;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background-color: white;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        th {
            background-color: #007acc;
            color: white;
            font-weight: 600;
        }

        tr:hover {
            background-color: #f5f5f5;
        }

        .rank-1 {
            background-color: #fff3cd;
        }

        .achieved {
            color: #28a745;
            font-weight: bold;
        }

        .not-achieved {
            color: #dc3545;
            font-weight: bold;
        }

        .features-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 20px;
        }

        .feature-item {
            background-color: #f9f9f9;
            padding: 10px 15px;
            border-radius: 4px;
            border-left: 3px solid #007acc;
        }

        .config-section {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }

        .config-item {
            margin-bottom: 10px;
        }

        .config-label {
            font-weight: 600;
            color: #555;
        }

        .config-value {
            color: #333;
            margin-left: 10px;
        }

        footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #eee;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }

        section {
            margin-bottom: 40px;
        }
        """

    def _generate_rankings_table(self, rankings: List[Any]) -> str:
        """Generate rankings table HTML."""
        if not rankings:
            return "<p>No rankings available</p>"

        html = "<table><thead><tr>"
        html += "<th>Rank</th>"
        html += "<th>Model</th>"
        html += "<th>Composite Score</th>"
        html += "<th>Objectives Achieved</th>"
        html += "</tr></thead><tbody>"

        for ranking in rankings:
            n_achieved = sum(1 for obj in ranking.objective_scores if obj.achieved)
            n_total = len(ranking.objective_scores)

            row_class = "rank-1" if ranking.rank == 1 else ""

            html += f"<tr class='{row_class}'>"
            html += f"<td>{ranking.rank}</td>"
            html += f"<td><strong>{ranking.model_name}</strong></td>"
            html += f"<td>{ranking.composite_score:.4f}</td>"
            html += f"<td>{n_achieved}/{n_total}</td>"
            html += "</tr>"

        html += "</tbody></table>"
        return html

    def _generate_objectives_table(self, best_model: Any) -> str:
        """Generate objectives table for best model."""
        html = "<table><thead><tr>"
        html += "<th>Objective</th>"
        html += "<th>Value</th>"
        html += "<th>Normalized</th>"
        html += "<th>Weight</th>"
        html += "<th>Weighted Score</th>"
        html += "<th>Achieved</th>"
        html += "</tr></thead><tbody>"

        for obj in best_model.objective_scores:
            achieved_class = "achieved" if obj.achieved else "not-achieved"
            achieved_text = "Yes" if obj.achieved else "No"

            value_str = f"{obj.value:.4f}" if not np.isnan(obj.value) else "N/A"

            html += "<tr>"
            html += f"<td><strong>{obj.name}</strong></td>"
            html += f"<td>{value_str}</td>"
            html += f"<td>{obj.normalized_score:.3f}</td>"
            html += f"<td>{obj.weight:.2f}</td>"
            html += f"<td>{obj.weighted_score:.3f}</td>"
            html += f"<td class='{achieved_class}'>{achieved_text}</td>"
            html += "</tr>"

        html += "</tbody></table>"
        return html

    def _generate_regime_stats_table(self, result: ExperimentResult) -> str:
        """Generate regime statistics table."""
        best_model = result.get_best_model()
        if not best_model:
            return "<p>No model available</p>"

        # path_structure_stats only exists on RegimeEvaluationResult, not SupervisedEvaluationResult
        path_stats = getattr(result.evaluation_result, "path_structure_stats", None)
        if path_stats is None or (hasattr(path_stats, "empty") and path_stats.empty):
            return "<p>No regime statistics available (supervised experiment)</p>"

        model_stats = path_stats[path_stats["algo"] == best_model.model_name]

        if len(model_stats) == 0:
            return "<p>No regime statistics available</p>"

        html = "<table><thead><tr>"
        html += "<th>Regime</th>"
        html += "<th>Observations</th>"
        html += "<th>Mean Duration</th>"
        html += "<th>Efficiency Ratio</th>"
        html += "<th>Volatility</th>"
        html += "<th>Autocorrelation</th>"
        html += "</tr></thead><tbody>"

        for _, row in model_stats.iterrows():
            regime = row.get("label", row.get("regime", "N/A"))
            html += "<tr>"
            html += f"<td><strong>{regime}</strong></td>"
            html += f"<td>{row.get('n_observations', 0):.0f}</td>"
            html += f"<td>{row.get('mean_duration', 0):.1f}</td>"
            html += f"<td>{row.get('efficiency_ratio', 0):.3f}</td>"
            html += f"<td>{row.get('volatility', 0):.4f}</td>"
            html += f"<td>{row.get('autocorrelation_lag1', 0):.3f}</td>"
            html += "</tr>"

        html += "</tbody></table>"
        return html

    def _generate_features_list(self, features: List[str]) -> str:
        """Generate features list HTML."""
        html = "<div class='features-list'>"
        for feature in features:
            html += f"<div class='feature-item'>{feature}</div>"
        html += "</div>"
        return html

    def _generate_config_section(self, config: Dict[str, Any]) -> str:
        """Generate configuration section HTML."""
        html = "<div class='config-section'>"

        # Model configuration
        if "model" in config:
            model_config = config["model"]
            html += "<div class='config-item'>"
            html += "<span class='config-label'>Model Type:</span>"
            html += (
                f"<span class='config-value'>{model_config.get('type', 'N/A')}</span>"
            )
            html += "</div>"

            if "variants" in model_config:
                html += "<div class='config-item'>"
                html += "<span class='config-label'>Variants:</span>"
                html += f"<span class='config-value'>{', '.join(model_config['variants'])}</span>"
                html += "</div>"

            if "n_states_range" in model_config:
                html += "<div class='config-item'>"
                html += "<span class='config-label'>N States Range:</span>"
                html += f"<span class='config-value'>{model_config['n_states_range']}</span>"
                html += "</div>"

        # Feature engineering
        if "feature_engineering" in config:
            fe_config = config["feature_engineering"]
            if "external_function" in fe_config:
                ext_fn = fe_config["external_function"]
                html += "<div class='config-item'>"
                html += "<span class='config-label'>Feature Function:</span>"
                html += f"<span class='config-value'>{ext_fn.get('module', '')}.{ext_fn.get('function', '')}</span>"
                html += "</div>"

        # Objectives
        if "objectives" in config:
            obj_config = config["objectives"]
            if "primary" in obj_config:
                n_objectives = len(obj_config["primary"])
                html += "<div class='config-item'>"
                html += "<span class='config-label'>Primary Objectives:</span>"
                html += f"<span class='config-value'>{n_objectives}</span>"
                html += "</div>"

        html += "</div>"
        return html


def generate_report(result: ExperimentResult, output_path: Optional[str] = None) -> str:
    """
    Generate HTML report (convenience function).

    Parameters
    ----------
    result : ExperimentResult
        Experiment result
    output_path : str, optional
        Output file path

    Returns
    -------
    str
        Path to generated report
    """
    generator = ReportGenerator()
    return generator.generate_report(result, output_path)
