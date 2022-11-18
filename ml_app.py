# Build a machine learning application: The iris dataset
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
np.random.seed(0)
# from sklearn.model_selection import cross_validate

iris = load_iris()
# separate the data into features and target
features = pd.DataFrame(iris.data, columns=iris.feature_names)
target = pd.Series(iris.target)

    
    
# split the data into train and test
x_train, x_, y_train, y_ = train_test_split(features, target, test_size=0.4, random_state=101)
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.5, random_state=101)
print("x_train.shape", x_train.shape, "y_train.shape", y_train.shape)
print("x_cv.shape", x_cv.shape, "y_cv.shape", y_cv.shape)
print("x_test.shape", x_test.shape, "y_test.shape", y_test.shape)


class StreamlitApp:

    def __init__(self):
        self.model = RandomForestClassifier(criterion='entropy', max_depth=5, max_features='auto', n_jobs=-1)

    def train_data(self):
        self.model.fit(x_train, y_train)
        return self.model
    
#     def cross_validation(model, _X, _y, _cv=5):
#       _scoring = ['accuracy', 'precision', 'recall', 'f1']
#       results = cross_validate(estimator=model,
#                                X=_X,
#                                y=_y,
#                                cv=_cv,
#                                scoring=_scoring,
#                                return_train_score=True)
      
#       return {"Training Accuracy scores": results['train_accuracy'],
#               "Mean Training Accuracy": results['train_accuracy'].mean()*100,
#               "Mean Training Precision": results['train_precision'].mean(),
#               "Mean Training Recall": results['train_recall'].mean(),
#               "Mean Training F1 Score": results['train_f1'].mean(),
#               "Validation Accuracy scores": results['test_accuracy'],
#               "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
#               "Mean Validation Precision": results['test_precision'].mean(),
#               "Mean Validation Recall": results['test_recall'].mean(),
#               "Mean Validation F1 Score": results['test_f1'].mean()
#               }

    def construct_sidebar(self):

        cols = [col for col in features.columns]

        st.sidebar.markdown(
            '<p class="header-style">Iris Data Classification</p>',
            unsafe_allow_html=True
        )
        sepal_length = st.sidebar.selectbox(
            f"Select {cols[0]}",
            sorted(features[cols[0]].unique())
        )

        sepal_width = st.sidebar.selectbox(
            f"Select {cols[1]}",
            sorted(features[cols[1]].unique())
        )

        petal_length = st.sidebar.selectbox(
            f"Select {cols[2]}",
            sorted(features[cols[2]].unique())
        )

        petal_width = st.sidebar.selectbox(
            f"Select {cols[3]}",
            sorted(features[cols[3]].unique())
        )
        values = [sepal_length, sepal_width, petal_length, petal_width]

        return values

    def plot_pie_chart(self, probabilities):
        fig = go.Figure(
            data=[go.Pie(
                    labels=list(iris.target_names),
                    values=probabilities[0]
            )]
        )
        fig = fig.update_traces(
            hoverinfo='label+percent',
            textinfo='value',
            textfont_size=15
        )
        return fig

    def construct_app(self):

        self.train_data()
        values = self.construct_sidebar()

        values_to_predict = np.array(values).reshape(1, -1)

        prediction = self.model.predict(values_to_predict)
        prediction_str = iris.target_names[prediction[0]]
        probabilities = self.model.predict_proba(values_to_predict)

        st.markdown(
            """
            <style>
            .header-style {
                font-size:25px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <style>
            .font-style {
                font-size:20px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="header-style"> Iris Dataset Predictions </p>',
            unsafe_allow_html=True
        )

        column_1, column_2 = st.columns(2)
        column_1.markdown(
            f'<p class="font-style" >Prediction </p>',
            unsafe_allow_html=True
        )
        column_1.write(f"{prediction_str}")

        column_2.markdown(
            '<p class="font-style" >Probability </p>',
            unsafe_allow_html=True
        )
        column_2.write(f"{probabilities[0][prediction[0]]}")

        fig = self.plot_pie_chart(probabilities)
        st.markdown(
            '<p class="font-style" >Probability Distribution</p>',
            unsafe_allow_html=True
        )
        st.plotly_chart(fig, use_container_width=True)

        return self


sa = StreamlitApp()
sa.construct_app()