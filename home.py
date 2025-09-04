import streamlit as st

def app():
    # Main title and subtitle
    st.title("🏠 HomeWorth Philly")
    st.markdown("#### Know Your Property’s Worth in Seconds")

    # Divider line
    st.markdown("---")

    # Main description
    st.markdown("""
    Welcome! **HomeMatch** helps you predict fair and accurate home prices using:

    - Property and zoning information  
    - Neighborhood insights and planning context  
    - Real estate market trends

    **Objective**:  
    Minimize pricing errors and prevent both **overpricing** and **undervaluing**.

    ---

    **To get started**, choose a page from the sidebar on the left!
    """)

if __name__ == "__main__":
    app()
