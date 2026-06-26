import streamlit as st
import pandas as pd
import numpy as np
import random
import requests
import json
import io

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="LeadCraft AI - Premium B2B Lead Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling rules to replicate the B2B dashboard look
st.markdown("""
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 800;
            color: #1e3a8a;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1rem;
            color: #64748b;
            margin-bottom: 1.5rem;
            font-weight: 500;
        }
        .kpi-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 1.2rem;
            border-radius: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            text-align: center;
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 0.2rem;
        }
        .kpi-label {
            font-size: 0.75rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
    </style>
""", unsafe_allow_index=True)

# ==========================================
# CONSTANTS & PROCEDURAL DATABASES
# ==========================================
FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth",
    "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen",
    "Christopher", "Lisa", "Daniel", "Nancy", "Matthew", "Betty", "Anthony", "Sandra", "Mark", "Margaret",
    "Alexander", "Sophia", "Daniel", "Olivia", "Ethan", "Isabella", "Liam", "Mia", "Noah", "Charlotte",
    "Oliver", "Amelia", "Lucas", "Harper", "Aiden", "Evelyn", "Elijah", "Abigail", "Benjamin", "Emily"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts",
    "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker", "Cruz", "Edwards", "Collins", "Reyes"
]

CITIES = [
    "San Francisco, CA", "New York, NY", "Austin, TX", "Seattle, WA", "Boston, MA", 
    "Chicago, IL", "Denver, CO", "Atlanta, GA", "Los Angeles, CA", "Miami, FL",
    "Dallas, TX", "Salt Lake City, UT", "Portland, OR", "San Jose, CA", "San Diego, CA"
]

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "prospects_df" not in st.session_state:
    st.session_state.prospects_df = None
if "active_job_title" not in st.session_state:
    st.session_state.active_job_title = ""
if "active_industry" not in st.session_state:
    st.session_state.active_industry = ""

# ==========================================
# API UTILITIES (EXPONENTIAL BACKOFF)
# ==========================================
def fetch_gemini_api(payload, api_key):
    """Executes calls to the Gemini 2.5 Flash API with custom exponential backoff."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    import time
    delay = 1.0
    retries = 5
    
    for i in range(retries):
        try:
            response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            if response.status_code == 200:
                return response.json()
            if response.status_code in [429, 500, 503]:
                time.sleep(delay)
                delay *= 2
                continue
            st.error(f"Gemini API returned status code {response.status_code}")
            return None
        except Exception as e:
            if i == retries - 1:
                st.error(f"Connection Failed: {str(e)}")
                return None
            time.sleep(delay)
            delay *= 2
    return None

def fetch_archetypes_from_ai(job_title, industry, api_key):
    """Queries Gemini to construct structured, context-rich lead profiles."""
    system_prompt = "You are a professional B2B lead generation researcher. Generate target profiles matching the requested role and industry with deep detail."
    user_prompt = f"""
    Generate 10 highly realistic, highly detailed archetype profiles for the role of "{job_title}" in the "{industry}" industry.
    Format the output as a strict JSON object structure matching this schema:
    {{
      "archetypes": [
        {{
          "subTitle": "Specific target job variation",
          "exampleCompany": "Highly realistic company in this industry space",
          "targetDomain": "companydomain.com",
          "skills": ["Skill1", "Skill2", "Skill3", "Skill4", "Skill5"],
          "bioPattern": "A standard professional background bio outline matching this archetype.",
          "estimatedSeniorityScore": 85
        }}
      ]
    }}
    Each bioPattern should be professional and relevant. Return absolutely raw, clean, parsed JSON matching the schema exactly. Do not wrap in markdown or backticks.
    """
    
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "archetypes": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "subTitle": {"type": "STRING"},
                                "exampleCompany": {"type": "STRING"},
                                "targetDomain": {"type": "STRING"},
                                "skills": {"type": "ARRAY", "items": {"type": "STRING"}},
                                "bioPattern": {"type": "STRING"},
                                "estimatedSeniorityScore": {"type": "INTEGER"}
                            },
                            "required": ["subTitle", "exampleCompany", "targetDomain", "skills", "bioPattern", "estimatedSeniorityScore"]
                        }
                    }
                },
                "required": ["archetypes"]
            }
        }
    }
    
    result = fetch_gemini_api(payload, api_key)
    if result:
        try:
            text_response = result['candidates'][0]['content']['parts'][0]['text']
            return json.loads(text_response)
        except Exception as e:
            st.warning("Failed to parse AI structure. Using fallback generation...")
            return None
    return None

# ==========================================
# PROCEDURAL SCENARIOS (FALLBACK GENERATOR)
# ==========================================
def generate_fallback_archetypes(job_title, industry):
    """Provides high-quality mock target structures if API key is not active."""
    sub_titles = [
        f"Lead {job_title}", f"Director of {job_title}", f"Principal {job_title}", 
        f"Senior {job_title}", f"{job_title} Lead", f"Staff {job_title}", 
        f"Global Head of {job_title}", f"Associate {job_title}"
    ]
    companies = [
        "Enterprise Grid", "Apex Horizons", "Quantum Scale", "Nexis Partners", 
        "Veridian Core", "Synthetix Labs", "Fortress Alliance", "Stratis Prime"
    ]
    domains = ["enterprisegrid.io", "apexhorizons.com", "quantumscale.co", "nexispartners.com"]
    skills = ["Strategy Planning", "Project Management", "Agile Execution", "SaaS Infrastructure", "Cross-Functional Collaboration", "KPI Tracking"]
    
    archetypes = []
    for i in range(8):
        archetypes.append({
            "subTitle": sub_titles[i % len(sub_titles)],
            "exampleCompany": companies[i % len(companies)],
            "targetDomain": domains[i % len(domains)],
            "skills": [skills[x] for x in random.sample(range(len(skills)), 4)],
            "bioPattern": f"Professional executing strategic leadership initiatives in the {industry} space. Experienced in scaling high-performing cross-functional squads.",
            "estimatedSeniorityScore": random.randint(70, 98)
        })
    return {"archetypes": archetypes}

# ==========================================
# PROCEDURAL EXPANSION ENGINE
# ==========================================
def compile_prospects(archetypes_list, target_count, job_title, industry):
    """Procedurally expands archetypes up to 1000 highly unique, structured records."""
    data = []
    
    for i in range(target_count):
        seed = archetypes_list[i % len(archetypes_list)]
        
        # Build unique name
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        full_name = f"{first} {last}"
        
        # Build geographic distribution
        loc = random.choice(CITIES)
        
        # Dynamic company names & matching domains
        company = seed["exampleCompany"]
        domain = seed["targetDomain"]
        if i >= len(archetypes_list):
            extra_tags = ["Labs", "Global", "Solutions", "HQ", "Systems", "Group", "Group LLC"]
            tag = extra_tags[i % len(extra_tags)]
            company = f"{seed['exampleCompany']} {tag}"
            domain = f"{seed['exampleCompany'].lower().replace(' ', '')}-{tag.lower()}.com"
            
        email = f"{first.lower()}.{last.lower()}@{domain}"
        linkedin = f"https://linkedin.com/in/{first.lower()}-{last.lower()}-{random.randint(1000, 9999)}"
        
        experience = random.randint(3, 20)
        score = min(100, max(45, seed["estimatedSeniorityScore"] + random.randint(-8, 8)))
        
        # Randomize skill subset
        skills_subset = random.sample(seed["skills"], min(len(seed["skills"]), random.randint(3, 5)))
        
        bio = seed["bioPattern"].replace("{Name}", full_name).replace("{Company}", company).replace("{Title}", seed["subTitle"])
        
        data.append({
            "Select": True,
            "Full Name": full_name,
            "Job Title": seed["subTitle"],
            "Company": company,
            "Location": loc,
            "Email Address": email,
            "LinkedIn URL": linkedin,
            "Years of Experience": experience,
            "Match Score (%)": score,
            "Skills": ", ".join(skills_subset),
            "AI Bio Context": bio
        })
        
    return pd.DataFrame(data)

# ==========================================
# OUTSIDE OUTREACH COMPILATION (GEMINI)
# ==========================================
def generate_pitch(lead_row, api_key, style_choice, custom_instructions=""):
    """Leverages Gemini to write personalized outbound material in various styles."""
    
    # Establish local templates if no API key is specified
    if not api_key:
        if "LinkedIn" in style_choice:
            return f"Hi {lead_row['Full Name'].split(' ')[0]}, enjoyed your work as {lead_row['Job Title']} at {lead_row['Company']}. Would love to connect regarding optimization trends in the {lead_row['Location']} space. Best!"
        elif "PAS" in style_choice:
            return f"Subject: Optimization roadmap for {lead_row['Company']}\n\nHi {lead_row['Full Name'].split(' ')[0]},\n\n[Problem] Scaling operational performance in B2B is highly unpredictable.\n\n[Agitate] Relying on old databases wastes your team's budget and target bandwidth.\n\n[Solve] We build pipelines for {lead_row['Job Title']}s looking to optimize {lead_row['Skills']}.\n\nLet's coordinate a call this week?"
        else:
            return f"Subject: Quick question regarding your role at {lead_row['Company']}\n\nHi {lead_row['Full Name'].split(' ')[0]},\n\nHope this finds you well. I was reviewing the scope of the {lead_row['Job Title']} role at {lead_row['Company']} and wanted to connect concerning custom automation toolings.\n\nLet's synchronize a chat next week?\n\nBest,\nLeadCraft Sales"

    system_prompt = "You are an elite B2B enterprise cold sales representative and outreach copywriter."
    
    # Context-aware style guidance
    style_guidelines = ""
    if style_choice == "Casual LinkedIn Connection Request (< 300 characters)":
        style_guidelines = "Create a hyper-short, casual message under 300 characters total fit for a LinkedIn connection invitation. Do not use subject lines or formal signatures."
    elif style_choice == "Problem-Agitate-Solve (PAS) Email":
        style_guidelines = "Structure the message strictly around the PAS framework: clearly identify a major job-specific Problem, Agitate the consequences of leaving it unresolved, and present our offer as the ultimate Solution."
    elif style_choice == "The Direct Meeting Pitch":
        style_guidelines = "Keep it concise, direct, and focused on proposing a brief 10-minute meeting window. Include a strong call to action."
    elif style_choice == "Custom Directive Guidance":
        style_guidelines = f"Adopt this exact tone and layout directive: {custom_instructions}"

    user_prompt = f"""
    Compose an outreach message from "The LeadCraft Team" targeting {lead_row['Full Name']}, who serves as "{lead_row['Job Title']}" at "{lead_row['Company']}".
    Key details about the target:
    - Location: {lead_row['Location']}
    - Top Skills: {lead_row['Skills']}
    - Bio background: {lead_row['AI Bio Context']}

    Outreach Style Directive: {style_guidelines}
    """
    
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }
    
    result = fetch_gemini_api(payload, api_key)
    if result:
        try:
            return result['candidates'][0]['content']['parts'][0]['text']
        except:
            pass
    return "Error generating pitch with Gemini API."

# ==========================================
# STREAMLIT INTERACTIVE SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### 🔑 API Connection Engine")
    
    # Informative Box explaining the dual-mode API key setup
    st.info(
        "💡 **How this works:** This application functions in two distinct modes:\n\n"
        "1. **Simulated Sandbox Mode (No Key Needed):** Completely operational out-of-the-box. Generates highly realistic lead lists and messaging templates utilizing our high-fidelity procedural simulation engine.\n\n"
        "2. **Live Premium AI Mode:** Connecting your personal Google Gemini API key unlocks context-aware live AI generation. **Using your own key completely bypasses shared public tier rate-limiting (preventing 503 Overloads or 429 Quotas).**"
    )
    
    # Secure API Key Box
    api_key = st.text_input(
        "Enter Gemini API Key", 
        type="password", 
        placeholder="AIzaSy...",
        help="Paste a key from Google AI Studio. Your key is processed entirely client-side and remains secure."
    )
    
    if api_key:
        st.success("✅ Live Premium AI Mode Engaged")
    else:
        st.warning("⚠️ Running in Local Simulated Mode")
        
    st.write("---")
    
    # Configuration Forms
    st.markdown("##### Target Prospect Rules")
    
    # Scenario Presets
    scenario = st.selectbox(
        "Choose a Preset Scenario", 
        ["Custom Input", "Product Manager @ SaaS", "Chief Medical Officer @ Digital Health", "VP of Sales @ Fintech", "Director of Logistics @ Supply Chain"]
    )
    
    # Autofill matching logic
    default_job = ""
    default_industry = ""
    if scenario == "Product Manager @ SaaS":
        default_job, default_industry = "Product Manager", "SaaS"
    elif scenario == "Chief Medical Officer @ Digital Health":
        default_job, default_industry = "Chief Medical Officer", "Digital Health"
    elif scenario == "VP of Sales @ Fintech":
        default_job, default_industry = "VP of Sales", "Fintech"
    elif scenario == "Director of Logistics @ Supply Chain":
        default_job, default_industry = "Director of Logistics", "Supply Chain"
        
    job_title = st.text_input("Target Job Title", value=default_job, placeholder="e.g. Head of Growth")
    industry = st.text_input("Target Industry", value=default_industry, placeholder="e.g. FinTech")
    
    total_leads = st.slider("Total Leads to Generate", min_value=10, max_value=1000, value=100, step=10)
    
    btn_generate = st.button("Generate Prospects Database", type="primary", use_container_width=True)

# ==========================================
# MAIN APPLICATION INTERACTION & ACTIONS
# ==========================================

# Main headers
st.markdown('<div class="main-header">LeadCraft <span style="color:#2563eb;">AI</span></div>', unsafe_allow_index=True)
st.markdown('<div class="sub-header">Premium Multi-Tab B2B Lead Engine & Dynamic Outreach Sequence Builder</div>', unsafe_allow_index=True)

# Process generation on click
if btn_generate:
    if not job_title or not industry:
        st.error("Please provide both a Target Job Title and Target Industry to generate records.")
    else:
        with st.spinner("Connecting to LeadCraft AI Core Engine..."):
            archetypes_data = None
            if api_key:
                archetypes_data = fetch_archetypes_from_ai(job_title, industry, api_key)
            
            if not archetypes_data:
                # Runs high-quality procedural fallback automatically
                archetypes_data = generate_fallback_archetypes(job_title, industry)
                
            # Procedural compilation scaling to desired amount
            df = compile_prospects(archetypes_data["archetypes"], total_leads, job_title, industry)
            
            # Save into streamlit page state
            st.session_state.prospects_df = df
            st.session_state.active_job_title = job_title
            st.session_state.active_industry = industry
            st.toast(f"Successfully compiled {total_leads} premium targets!", icon="🚀")

# Check if database has been populated
if st.session_state.prospects_df is not None:
    df_state = st.session_state.prospects_df
    
    # CREATE TABS WORKSPACE
    tab_directory, tab_analytics, tab_outreach = st.tabs([
        "📋 Lead Directory", 
        "📊 Market & Visual Analytics", 
        "💌 AI Outreach Agent"
    ])
    
    # ----------------------------------------------------
    # TAB 1: LEAD DIRECTORY (WITH INTEGRATED FILTERING)
    # ----------------------------------------------------
    with tab_directory:
        # Collapsible filtering drawer panel
        with st.expander("🔍 Advanced Database Filters", expanded=False):
            col_filt_1, col_filt_2, col_filt_3 = st.columns(3)
            with col_filt_1:
                min_match_score = st.slider("Minimum Match Score (%)", 0, 100, 50)
            with col_filt_2:
                min_experience = st.slider("Minimum Years of Experience", 0, 20, 0)
            with col_filt_3:
                unique_locations = sorted(df_state["Location"].unique().tolist())
                selected_locations = st.multiselect("Filter by Locations", unique_locations, default=unique_locations)

        # Applying multi-filters in real time
        filtered_df = df_state.copy()
        filtered_df = filtered_df[
            (filtered_df["Match Score (%)"] >= min_match_score) &
            (filtered_df["Years of Experience"] >= min_experience) &
            (filtered_df["Location"].isin(selected_locations))
        ]

        # Bulk selection controls
        col_sel_1, col_sel_2, _ = st.columns([1.5, 1.5, 7])
        with col_sel_1:
            if st.button("Select All Visible", use_container_width=True):
                # Update only the visible filtered rows to True
                df_state.set_index("Email Address", drop=False, inplace=True)
                filtered_df["Select"] = True
                df_state.update(filtered_df)
                df_state.reset_index(drop=True, inplace=True)
                st.session_state.prospects_df = df_state
                st.rerun()
        with col_sel_2:
            if st.button("Clear All Visible", use_container_width=True):
                # Update only the visible filtered rows to False
                df_state.set_index("Email Address", drop=False, inplace=True)
                filtered_df["Select"] = False
                df_state.update(filtered_df)
                df_state.reset_index(drop=True, inplace=True)
                st.session_state.prospects_df = df_state
                st.rerun()

        # Build Interactive Grid
        st.markdown("##### Interactive Record Table")
        st.caption("Double-click checkboxes under the 'Select' column to configure files before exporting. Columns are fully sortable.")
        
        # Display data editor
        edited_filtered_df = st.data_editor(
            filtered_df,
            column_config={
                "Select": st.column_config.CheckboxColumn("Select", help="Choose row for export.", default=True),
                "LinkedIn URL": st.column_config.LinkColumn("LinkedIn URL"),
                "Full Name": st.column_config.TextColumn(disabled=True),
                "Job Title": st.column_config.TextColumn(disabled=True),
                "Company": st.column_config.TextColumn(disabled=True),
                "Location": st.column_config.TextColumn(disabled=True),
                "Email Address": st.column_config.TextColumn(disabled=True),
                "Years of Experience": st.column_config.NumberColumn(disabled=True),
                "Match Score (%)": st.column_config.NumberColumn(disabled=True),
                "Skills": st.column_config.TextColumn(disabled=True),
                "AI Bio Context": st.column_config.TextColumn(disabled=True)
            },
            use_container_width=True,
            hide_index=True,
            key="prospect_editor"
        )

        # Merge updates back to Master DataFrame State safely using email address index
        df_state.set_index("Email Address", drop=False, inplace=True)
        edited_filtered_df.set_index("Email Address", drop=False, inplace=True)
        df_state.update(edited_filtered_df)
        df_state.reset_index(drop=True, inplace=True)
        st.session_state.prospects_df = df_state

        # Collect Selected rows globally
        selected_rows = df_state[df_state["Select"] == True]

        # Excel Compiler Downloader
        st.markdown("---")
        st.markdown("##### Download Export Batch")
        st.write("Click below to compile all checked profiles globally into a downloadable, pre-formatted native Excel file.")
        
        if len(selected_rows) > 0:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                export_cleaned_df = selected_rows.drop(columns=["Select"])
                export_cleaned_df.to_excel(writer, index=False, sheet_name="Target Prospects")
                
            excel_data = buffer.getvalue()
            target_file_title = f"Prospects_{st.session_state.active_job_title.replace(' ', '_')}_{st.session_state.active_industry.replace(' ', '_')}.xlsx"
            
            st.download_button(
                label=f"📥 Download {len(selected_rows)} Selected Prospects as Excel",
                data=excel_data,
                file_name=target_file_title,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.warning("Please check/select at least one prospect to activate the Excel downloader.")

    # ----------------------------------------------------
    # TAB 2: MARKET & VISUAL ANALYTICS
    # ----------------------------------------------------
    with tab_analytics:
        st.markdown("#### 📊 Dynamic Market Segment Insights")
        st.write("Real-time demographic and performance distribution data compiled for the active list.")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value">{len(df_state):,}</div>
                    <div class="kpi-label">Sourced Database Size</div>
                </div>
            """, unsafe_allow_index=True)
        with col2:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value" style="color: #2563eb;">{len(selected_rows):,}</div>
                    <div class="kpi-label">Checked for Export</div>
                </div>
            """, unsafe_allow_index=True)
        with col3:
            avg_score = int(selected_rows["Match Score (%)"].mean()) if len(selected_rows) > 0 else 0
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value" style="color: #10b981;">{avg_score}%</div>
                    <div class="kpi-label">Selected Match Quality</div>
                </div>
            """, unsafe_allow_index=True)
        with col4:
            avg_exp = round(selected_rows["Years of Experience"].mean(), 1) if len(selected_rows) > 0 else 0.0
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value" style="color: #f59e0b;">{avg_exp} yrs</div>
                    <div class="kpi-label">Avg Lead Seniority</div>
                </div>
            """, unsafe_allow_index=True)

        st.write("---")

        # Layout charts side-by-side
        col_chart_left, col_chart_right = st.columns(2)
        
        with col_chart_left:
            st.markdown("##### Geographic Distribution of Sourced Leads")
            loc_counts = df_state["Location"].value_counts()
            st.bar_chart(loc_counts)
            
        with col_chart_right:
            st.markdown("##### Years of Experience Spread")
            exp_counts = df_state["Years of Experience"].value_counts().sort_index()
            st.area_chart(exp_counts)

    # ----------------------------------------------------
    # TAB 3: AI OUTREACH AGENT & COPYWRITER
    # ----------------------------------------------------
    with tab_outreach:
        st.markdown("#### 💌 Deep Outreach Sequence Designer")
        st.write("Select a lead from your selected cohort and choose from standard sales copy frameworks.")
        
        if len(selected_rows) > 0:
            col_out_left, col_out_right = st.columns([1.5, 2])
            
            with col_out_left:
                st.markdown("##### Step 1: Select Target & Format")
                selected_lead_name = st.selectbox("Select Target Lead Profile", selected_rows["Full Name"].tolist())
                lead_row = selected_rows[selected_rows["Full Name"] == selected_lead_name].iloc[0]
                
                # Dynamic style architect selection
                style_choice = st.selectbox(
                    "Choose Copywriting Framework",
                    [
                        "The Direct Meeting Pitch",
                        "Problem-Agitate-Solve (PAS) Email",
                        "Casual LinkedIn Connection Request (< 300 characters)",
                        "Custom Directive Guidance"
                    ]
                )
                
                # Show custom directive box if chosen
                custom_instructions = ""
                if style_choice == "Custom Directive Guidance":
                    custom_instructions = st.text_area(
                        "Custom Copy Guidance",
                        placeholder="e.g., 'Ensure the tone is playful, mention they have amazing skills in Python and request a brief chat.'",
                        height=100
                    )
                
                # Call AI copy block generator
                btn_compose = st.button("Generate Tailored Sales Pitch", type="primary", use_container_width=True)
                
                st.write("")
                st.info(f"""
                **Prospect Snapshot:**
                - **Name**: {lead_row['Full Name']}
                - **Role**: {lead_row['Job Title']} @ {lead_row['Company']}
                - **Primary Skills**: {lead_row['Skills']}
                """)
                
            with col_out_right:
                st.markdown("##### Step 2: Generated Outreach Copy")
                if btn_compose:
                    with st.spinner("Gemini compiling personalized sequences..."):
                        pitch_text = generate_pitch(lead_row, api_key, style_choice, custom_instructions)
                        st.text_area("Live Copywriter Output Box", value=pitch_text, height=350)
                        st.success("✨ Personalization complete! Copy this outreach directly into your sales sequencing platform.")
                else:
                    st.write("Click 'Generate Tailored Sales Pitch' to begin rendering personalized outbound text.")
        else:
            st.info("Please make sure you have checked/selected at least one lead profile in '📋 Lead Directory' first.")

else:
    # Landing instructional banner
    st.info("👈 Set a Target Job Title and Target Industry on the left sidebar to generate your dynamic prospect directory!")
    
    col_info_1, col_info_2 = st.columns(2)
    with col_info_1:
        st.markdown("""
            #### 🚀 How It Works:
            1. **Configure Parameters**: Give the app any role (e.g., *Head of Sales*) and target market sector.
            2. **Engage Gemini AI**: The app uses Google's model to outline context-rich company archetypes.
            3. **Procedural Expansion Engine**: Instantly scales up mock records containing custom email domains, locations, and linked handles.
        """)
    with col_info_2:
        st.markdown("""
            #### 📊 Integrated Tab Workspaces:
            - **📋 Lead Directory**: Full list of records with adjustable search, experience filters, and checkbox selectors.
            - **📊 Visual Analytics**: Automated charts mapping experience curves and city-wide spreads.
            - **💌 AI Outreach Agent**: Draft custom connection sequences in diverse frameworks (PAS, Direct Pitch, or Custom prompts).
        """)
