import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.subheader('Cache a Dataframe')
with st.echo():
    def return_df(filename):
        df = pd.read_csv(filename)
        st.write('Inside the load function')
        return df

    @st.memo(persist=True)
    def load_csv(filename):
        df = return_df(filename)
        return df

    df = load_csv('netflix.csv')
    st.write(df.head())

st.markdown('---')

st.subheader('Cache a Matplotlib figure')
with st.echo():
    @st.memo
    def make_plot(loc, scale):
        st.write(f'Cache miss with params, loc={loc}, scale={scale}')
        arr = np.random.normal(loc, scale, size=100)
        fig, ax = plt.subplots()
        ax.hist(arr, bins=20)
        return fig, ax

    loc = st.number_input('Center of distibution', min_value=-5.0, max_value=5.0, value=0.0, key='loc')
    scale = st.slider('Standard Deviation', min_value=1.0, max_value=5.0, value=2.0, key='scale')

    fig, ax = make_plot(loc, scale)
    st.pyplot(fig)

st.markdown('---')

st.subheader('Return Numpy Array')

with st.echo():
    # See https://github.com/streamlit/streamlit/issues/2924
    @st.memo
    def minimal_example():
        import numpy

        a = numpy.array([1, 2])  # this is fine
        b = a.tolist()  # this is also fine

        c = numpy.ndarray.tolist(a)  # this errors
        d = numpy.ndarray.sum(a)  # so does this

        return d

    d = minimal_example()
    st.write(d)

st.markdown('---')

st.subheader('Caching Session State?')

with st.echo():
    if 'foo' not in st.session_state:
        st.session_state.foo = 5

    @st.memo
    def return_session_state():
        return st.session_state

    x = return_session_state()
    st.write(x)

    # mutate session state
    st.session_state.foo = 10
    x = return_session_state()
    st.write(x)