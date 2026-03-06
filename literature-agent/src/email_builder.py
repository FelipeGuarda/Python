"""Build HTML digest email from papers grouped by topic."""

from collections import OrderedDict
from datetime import datetime

from jinja2 import Template

MONTHS_ES = [
    "", "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]

TEMPLATE_STR = """\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: Georgia, 'Times New Roman', serif; max-width: 700px;
             margin: 0 auto; padding: 20px; color: #2c2c2c; line-height: 1.6;">

  <h1 style="color: #1a5c2e; border-bottom: 2px solid #1a5c2e; padding-bottom: 8px;">
    {{ subject }}
  </h1>

  <p style="color: #666; font-size: 14px;">
    {{ total }} artículos encontrados en {{ sources }} fuentes.
  </p>

  {% for topic_name, topic_papers in papers_by_topic.items() %}
  <h2 style="color: #1a5c2e; margin-top: 30px; border-bottom: 1px solid #ccc;
             padding-bottom: 4px;">
    {{ topic_name }} ({{ topic_papers|length }})
  </h2>

  {% for paper in topic_papers %}
  <div style="margin-bottom: 20px; padding: 12px; background: #f9f9f6;
              border-left: 3px solid #1a5c2e;">
    <strong style="font-size: 15px;">
      [{{ loop.index }}] {{ paper.title }}
    </strong><br>
    <span style="font-size: 13px; color: #555;">
      {{ paper.authors }}{% if paper.date %} | {{ paper.date }}{% endif %}
       | {{ paper.source }}
    </span><br>
    {% if paper.doi %}
    <span style="font-size: 12px; color: #888;">DOI: {{ paper.doi }}</span><br>
    {% endif %}
    <em style="font-size: 14px;">{{ paper.summary }}</em><br>
    {% if paper.url %}
    <a href="{{ paper.url }}" style="color: #1a5c2e; font-size: 13px;">
      Ver artículo completo →
    </a>
    {% endif %}
  </div>
  {% endfor %}
  {% endfor %}

  <hr style="margin-top: 40px; border: none; border-top: 1px solid #ccc;">
  <p style="font-size: 12px; color: #999;">
    Generado automáticamente por el Agente de Literatura — Fundación Mar Adentro.
  </p>
</body>
</html>
"""


def build_html(papers, config):
    """Return (subject, html_body) tuple."""
    now = datetime.now()
    date_es = f"{now.day} de {MONTHS_ES[now.month]} {now.year}"
    prefix = config["email"].get("subject_prefix", "[FMA Literatura]")
    subject = f"{prefix} Semana del {date_es}"

    # Group by topic, preserving config order
    topic_order = [t["name"] for t in config["topics"]]
    papers_by_topic = OrderedDict()
    for name in topic_order:
        matching = [p for p in papers if p.get("topic") == name]
        if matching:
            papers_by_topic[name] = matching

    sources = len({p["source"] for p in papers})
    template = Template(TEMPLATE_STR)
    html = template.render(
        subject=subject,
        total=len(papers),
        sources=sources,
        papers_by_topic=papers_by_topic,
    )

    return subject, html
