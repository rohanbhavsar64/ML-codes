import streamlit as st
st.title('Get In Touch with me')


cf = """
<form action="https://formsubmit.co/bhavsarrohan5264@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Name" required>
     <input type="email" name="email" placeholder="Email" required>
     <textarea name="message" placeholder="Details of your problem"></textarea>
     <button type="submit">Send</button>
</form>"""

st.markdown(cf,unsafe_allow_html=True)