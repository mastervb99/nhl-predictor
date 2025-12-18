"""
PDF Export Utility
Generate downloadable PDF reports for game predictions
"""
from typing import Dict, List
from datetime import date
import io

# Try to import reportlab, fall back gracefully if not available
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


class PDFExporter:
    """Generate PDF reports for NHL predictions."""

    def __init__(self):
        if not HAS_REPORTLAB:
            raise ImportError("reportlab is required for PDF export. Install with: pip install reportlab")

        self.styles = getSampleStyleSheet()

        # Custom styles
        self.styles.add(ParagraphStyle(
            name='GameTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=6,
            spaceBefore=12,
        ))

    def generate_game_pdf(self, prediction: Dict) -> bytes:
        """
        Generate PDF report for a single game.

        Args:
            prediction: Dict with full prediction data

        Returns:
            PDF as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch)
        elements = []

        # Title
        home = prediction.get('home_team', {}).get('code', 'HOME')
        away = prediction.get('away_team', {}).get('code', 'AWAY')
        title = f"{away} @ {home} - Game Prediction"
        elements.append(Paragraph(title, self.styles['GameTitle']))
        elements.append(Paragraph(f"Generated: {date.today().isoformat()}", self.styles['Normal']))
        elements.append(Spacer(1, 12))

        # Win Probabilities
        elements.append(Paragraph("Win Probabilities", self.styles['SectionHeader']))
        mc = prediction.get('monte_carlo', {}).get('final', {})
        prob_data = [
            ['Team', 'Win %', 'ML Edge'],
            [home, f"{mc.get('home_win_prob', 0)*100:.1f}%", f"{prediction.get('edge', {}).get('moneyline', {}).get('home', {}).get('edge', 0)*100:+.1f}%"],
            [away, f"{mc.get('away_win_prob', 0)*100:.1f}%", f"{prediction.get('edge', {}).get('moneyline', {}).get('away', {}).get('edge', 0)*100:+.1f}%"],
        ]
        prob_table = Table(prob_data, colWidths=[1.5*inch, 1*inch, 1*inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        elements.append(prob_table)
        elements.append(Spacer(1, 12))

        # Score Prediction
        elements.append(Paragraph("Score Prediction", self.styles['SectionHeader']))
        poisson = prediction.get('poisson', {})
        score_data = [
            ['Metric', 'Home', 'Away', 'Total'],
            ['Expected Goals', f"{poisson.get('lambda_home', 0):.2f}", f"{poisson.get('lambda_away', 0):.2f}", f"{poisson.get('expected_total', 0):.1f}"],
            ['Most Likely Score', poisson.get('most_likely_score', 'N/A'), '', ''],
        ]
        score_table = Table(score_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        elements.append(score_table)
        elements.append(Spacer(1, 12))

        # Over/Under
        elements.append(Paragraph("Over/Under Analysis", self.styles['SectionHeader']))
        ou_line = prediction.get('odds', {}).get('over_under', 6.0)
        ou_prob = prediction.get('monte_carlo', {}).get('over_under', {}).get(ou_line, {})
        ou_data = [
            ['Line', 'Over %', 'Under %', 'Edge'],
            [f"{ou_line}", f"{ou_prob.get('over_prob', 0)*100:.1f}%", f"{ou_prob.get('under_prob', 0)*100:.1f}%",
             f"{prediction.get('edge', {}).get('over_under', {}).get('over', {}).get('edge', 0)*100:+.1f}%"],
        ]
        ou_table = Table(ou_data, colWidths=[1*inch, 1*inch, 1*inch, 1*inch])
        ou_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        elements.append(ou_table)
        elements.append(Spacer(1, 12))

        # 1st Period
        p1 = prediction.get('first_period', {})
        if p1:
            elements.append(Paragraph("1st Period Prediction", self.styles['SectionHeader']))
            p1_data = [
                ['Metric', 'Value'],
                ['Expected Total', f"{p1.get('expected_total', 0):.2f}"],
                ['Over 1.5 Prob', f"{p1.get('over_under', {}).get(1.5, {}).get('over_prob', 0)*100:.1f}%"],
                ['Most Likely Score', p1.get('most_likely_score', 'N/A')],
            ]
            p1_table = Table(p1_data, colWidths=[2*inch, 1.5*inch])
            p1_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
            ]))
            elements.append(p1_table)
            elements.append(Spacer(1, 12))

        # Props
        props = prediction.get('props', {})
        if props:
            elements.append(Paragraph("Prop Bets", self.styles['SectionHeader']))
            props_data = [
                ['Prop', 'Probability', 'vs League Avg'],
                ['GIFT (Goal in First 10 Min)', f"{props.get('gift', {}).get('gift_prob', 0)*100:.1f}%",
                 f"{props.get('gift', {}).get('vs_league', 0)*100:+.1f}%"],
                ['1+ SOG First 2 Min', f"{props.get('sog_2min', {}).get('either_team_sog_prob', 0)*100:.1f}%", ''],
            ]
            props_table = Table(props_data, colWidths=[2.5*inch, 1.25*inch, 1.25*inch])
            props_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
            ]))
            elements.append(props_table)
            elements.append(Spacer(1, 12))

        # Recommendations
        edge = prediction.get('edge', {})
        top_plays = edge.get('top_plays', [])
        if top_plays:
            elements.append(Paragraph("Recommendations", self.styles['SectionHeader']))
            for i, play in enumerate(top_plays[:3], 1):
                text = f"{i}. {play.get('type', '').upper()} {play.get('side', '').upper()} - Edge: {play.get('edge', 0)*100:+.1f}% ({play.get('confidence', '')})"
                elements.append(Paragraph(text, self.styles['Normal']))

        # Build PDF
        doc.build(elements)
        return buffer.getvalue()

    def generate_overview_pdf(self, predictions: List[Dict]) -> bytes:
        """
        Generate overview PDF with all daily games.

        Args:
            predictions: List of prediction dicts

        Returns:
            PDF as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch)
        elements = []

        # Title
        elements.append(Paragraph(f"NHL Predictions - {date.today().isoformat()}", self.styles['GameTitle']))
        elements.append(Spacer(1, 12))

        # Summary table
        summary_data = [['Game', 'Home Win %', 'Away Win %', 'Total', 'Best Bet']]

        for pred in predictions:
            home = pred.get('home_team', {}).get('code', 'HOME')
            away = pred.get('away_team', {}).get('code', 'AWAY')
            mc = pred.get('monte_carlo', {}).get('final', {})

            top_play = pred.get('edge', {}).get('top_plays', [{}])[0] if pred.get('edge', {}).get('top_plays') else {}
            best_bet = f"{top_play.get('type', 'PASS')} {top_play.get('side', '')}"

            summary_data.append([
                f"{away} @ {home}",
                f"{mc.get('home_win_prob', 0)*100:.1f}%",
                f"{mc.get('away_win_prob', 0)*100:.1f}%",
                f"{pred.get('poisson', {}).get('expected_total', 0):.1f}",
                best_bet,
            ])

        summary_table = Table(summary_data, colWidths=[1.5*inch, 1*inch, 1*inch, 0.75*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        elements.append(summary_table)

        doc.build(elements)
        return buffer.getvalue()


def export_game_pdf(prediction: Dict) -> bytes:
    """Convenience function for single game export."""
    exporter = PDFExporter()
    return exporter.generate_game_pdf(prediction)


def export_overview_pdf(predictions: List[Dict]) -> bytes:
    """Convenience function for overview export."""
    exporter = PDFExporter()
    return exporter.generate_overview_pdf(predictions)
