
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn import tree
from IPython.display import Image


def run_decision_tree():
    iris = datasets.load_iris()
    features = iris.data
    target = iris.target

    print("Features: 1. sepal length 2. sepal width 3. petal length 4. petal width")
    print(features)
    print("Classes: 1. Iris Setosa 2. Iris Versicolour, 3. Iris Virginica")
    print(target)

    decision_tree = DecisionTreeClassifier(random_state=0)
    model = decision_tree.fit(features, target)  # train the algorithm with features and target values

    print("Prediction:")
    observation = [[5, 4, 3, 2]]  # Predict observation's class
    print(model.predict(observation))
    print(model.predict_proba(observation))

    dot_data = tree.export_graphviz(decision_tree,
                                    out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names
                                    )
    graph = pydotplus.graph_from_dot_data(dot_data)  # Show graph
    image = Image(graph.create_png())

    with open("decision_tree_image.png", "wb") as f_out:
        f_out.write(image.data)


if __name__ == '__main__':
    run_decision_tree()
