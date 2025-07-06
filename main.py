import streamlit as st
import langchain_helper

st.markdown(
    """
    <style>
    
        .cafe-card {
            background-color: #ffccdd;  /* Soft pink card */
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 12px;
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
            color: #C71585 !important;
        }
        .markdown {
            color: #C71585 !important; /* Dark pink (MediumVioletRed) */
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        body {
            background-color: #ffe6f0 !important; /* Baby pink */
        }
        .stApp {
            background-color: #ffe6f0 !important; /* Fallback for main container */
        }
        
        .title {
            color: #c71585 !important;  /* Dark pink (hex for MediumVioletRed) */
            font-size: 40px;
            font-weight: bold;
            text-align: center;
        }
        
        .subheader {
            color: #c71585 !important;  /* Dark pink (hex for MediumVioletRed) */
            font-size: 40px;
            font-weight: bold;
            text-align: center;
        }
        
        label[data-baseweb="select"] {
            color: #C71585 !important;
            font-weight: 600;
            font-size: 18px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">CuppaQuest</h1>', unsafe_allow_html=True)

st.markdown(
    """ 
    <div style='text-align: center; color: #c71585; font-size: 20px;'>
        Find your next Cafe Quest here!<br><br>
        <strong>There's <span style='background: linear-gradient(90deg, violet, pink); -webkit-background-clip: text; color: transparent;'>so many cute cafes in Chennai</span> for you to visit!</strong>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """<div style='color:#C71585; font-size:18px; font-weight:600; margin-bottom:6px;'>Location:</div>""",
    unsafe_allow_html=True
)

location = st.selectbox(
    "label:",
    options = ["Select a location",
    "Adyar",
    "Alwarpet",
    "Anna Nagar",
    "Ashok Nagar",
    "Besant Nagar",
    "ECR",
    "Egmore",
    "Guindy",
    "Kotturpuram",
    "Mylapore",
    "Nandanam",
    "Neelankarai",
    "Nungambakkam",
    "OMR",
    "Porur",
    "R.A. Puram",
    "Royapettah",
    "T. Nagar",
    "Teynampet",
    "Thiruvanmiyur",
    "Thousand Lights",
    "Triplicane",
    "Velachery"
],
label_visibility="collapsed")

if location and location != "Select a location":
    cafes = langchain_helper.get_cafe_information(location)
    st.subheader(f"Cafes in {location.title()}")
    for name, description in cafes:
            with st.container():
                st.markdown(f"""
                    <div class="cafe-card">
                        <h4 style="margin-bottom: 8px;">{name}</h4>
                        <h5 style="margin-bottom: 8px;">{description}</h5>
                    </div>
                """, unsafe_allow_html=True)

elif location == "Select a location":
    st.info("Please select a location to explore cute cafes!")

else:
        st.warning("No cafes found for that location.")

question = st.text_input("Enter a question")

if question:
    with st.spinner("Thinking..."):
        response = langchain_helper.q_and_a(question, location)
        st.markdown(f"**Your Question:** {question}")
        st.markdown(f"**Answer:** {response['answer']}")