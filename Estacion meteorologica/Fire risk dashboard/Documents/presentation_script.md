# Fire Risk Dashboard — Bosque Pehuén
## Video Presentation Script (2-5 Minutes)

**Total Duration: 4 minutes 30 seconds**

---

### **SEGMENT 1: Problem Introduction [0:00–0:45] (45 seconds)**

**[VISUAL: Title slide with Bosque Pehuén landscape image]**

"Hello, I'm Felipe Guarda. This presentation demonstrates a data visualization project focused on wildfire risk assessment in Bosque Pehuén, a privately protected area in southern Chile managed by Fundación Mar Adentro.

**[VISUAL: Map of Chile highlighting the study area]**

Wildfire is a critical threat to ecosystems, infrastructure, and communities in Chile's Araucanía region. The challenge isn't just detecting fires after they occur—it's predicting risk *before* they happen so land managers and communities can prepare.

The core problem is: **How can we visualize and communicate fire risk to stakeholders in real time, using open data and interactive tools?**"

---

### **SEGMENT 2: Previous Work & Methodology [0:45–1:45] (60 seconds)**

**[VISUAL: Show slides describing fire indices and data sources]**

"To address this, I reviewed literature on fire-danger rating systems used globally—from Canada's CFFDRS to the U.S. National Fire Danger Rating System, down to Chile and Argentina's regional approaches.

**[VISUAL: Table showing fire risk variables]**

I selected four key meteorological variables that drive fire danger:
- **Temperature**: Higher temperatures increase flammability
- **Relative Humidity**: Inverse relationship—dry air accelerates fire spread  
- **Wind Speed**: Critical for propagation and intensity
- **Days Without Rain**: A drought indicator using a precipitation threshold of 2 millimeters

**[VISUAL: Open-Meteo API interface]**

For data, I integrated the Open-Meteo API—a freely available global weather platform providing hourly forecasts and historical data under a Creative Commons license. This ensures reproducibility and accessibility.

**[VISUAL: Risk scoring table]**

Each variable receives a partial score from 0 to 100, weighted according to Chilean fire-danger guidelines. The dashboard computes risk scores for afternoon peak-fire hours—14:00 to 16:00 local time—when conditions are most critical."

---

### **SEGMENT 3: What Was Accomplished [1:45–3:15] (90 seconds)**

**[VISUAL: Show the live Streamlit dashboard interface]**

"I built an interactive web-based dashboard using Streamlit, an open-source Python framework for data visualization. Here's what it does:

**[VISUAL: Demonstrate the daily polar plot]**

**First, a polar plot** shows daily risk variables. Each axis represents a different factor—temperature, humidity, wind, and drought duration. The plot fills with a color that reflects the overall risk level: green for low, yellow for moderate, orange for high, and red for extreme. This instantly communicates how variables contribute to risk.

**[VISUAL: Demonstrate the score summary table]**

**Second, a risk summary table** displays the exact numerical scores for each variable and the total risk index from 0 to 100, with risk categories clearly labeled.

**[VISUAL: Demonstrate the wind compass]**

**Third, a wind compass** shows wind direction and speed. The wedge indicates where wind is coming from—critical information for fire spread prediction. Wind speed is encoded both in the wedge length and color intensity.

**[VISUAL: Demonstrate the forecast time-series chart]**

**Fourth, a 14-day risk forecast** displays as a bar chart. Each bar is color-coded by risk level, allowing stakeholders to identify high-risk periods at a glance.

**[VISUAL: Demonstrate the regional wind map]**

**Finally, a regional map** uses Pydeck to visualize wind flow patterns across the Araucanía region. Streamlines show wind direction and intensity with color gradients—cyan for calm, progressing to red for extreme winds. Bosque Pehuén is highlighted, anchoring the visualization in geographic context.

**[VISUAL: Show the modular code structure]**

Behind the scenes, the codebase is organized into six specialized modules: a configuration module for parameters, a data fetcher for API integration, a risk calculator for scoring logic, visualization functions for charts, and map utilities. This modular design makes the system maintainable, testable, and easy to extend."

---

### **SEGMENT 4: Results & Impact [3:15–4:15] (60 seconds)**

**[VISUAL: Show sample dashboard screenshots with different risk scenarios]**

"The dashboard successfully demonstrates how meteorological data can be translated into actionable fire-risk intelligence. Here are key results:

**Real-time and Forecast Capability**: The system fetches current weather data and generates 14-day forecasts continuously, ensuring decision-makers always have up-to-date information.

**Clear Risk Communication**: By combining multiple visualization types—polar plots, time-series charts, and wind maps—the dashboard communicates complex fire science in an intuitive, non-technical format accessible to landowners, park rangers, and emergency managers.

**Reproducibility**: All code is open-source, using only freely licensed data and libraries. This means the approach can be adapted to other protected areas in Chile or globally.

**[VISUAL: Show the GitHub repository or documentation]**

The system includes full documentation, environment specifications, and sample datasets—enabling others to replicate or extend the work.

**Environmental Context**: By focusing on Bosque Pehuén, a flagship conservation property, the project demonstrates how data visualization can enhance environmental stewardship and community preparedness."

---

### **SEGMENT 5: Conclusion [4:15–4:30] (15 seconds)**

**[VISUAL: Return to title slide or project summary]**

"In summary, this project integrates fire science, open data, and interactive visualization to create a practical tool for wildfire risk management. The dashboard is live, reproducible, and ready for deployment in conservation and land-management contexts.

Thank you for watching. I'm happy to answer any questions."

---

## **Visual Assets You Should Prepare for the Video:**

1. **Title slide** — Project name, your name, institution, date
2. **Map of Chile** — Highlighting Bosque Pehuén's location in Araucanía
3. **Screenshot of the live dashboard** — Show multiple sections (polar plot, table, compass, forecast chart, map)
4. **Demo of interactivity** — Click through date selectors, show how the dashboard updates
5. **Code architecture diagram** — Show how the modules connect
6. **Conclusion slide** — Key takeaways and contact information

---

## **Tips for Recording:**

- **Pacing**: Speak clearly and at a moderate pace; 4 minutes 30 seconds is achievable with natural pauses.
- **Screen sharing**: Use screen recording software to capture live dashboard interactions.
- **Emphasis**: Pause briefly before introducing each major visualization or finding—allows viewers to absorb.
- **Accessibility**: Add subtitles or captions for clarity.
- **Audio**: Record in a quiet environment with clear microphone input.

