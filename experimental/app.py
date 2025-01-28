import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

st.write("""
# Stage Lens
""")

df = pd.DataFrame(np.random.randn(50, 20), columns=("col %d" % i for i in range(20)))
st.dataframe(df) # Same as st.write(df)

arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

st.pyplot(fig)

# streamlit run app.py --server.port 20000

