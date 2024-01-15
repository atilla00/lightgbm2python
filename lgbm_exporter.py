def _get_attributes(model, categoric_mappings):
    dump = model._Booster.dump_model()
    trees = [element["tree_structure"] for element in dump["tree_info"]]
    n_classes = getattr(model, "n_classes_", -1)
    feature_names = model.feature_name_
    return trees, categoric_mappings, n_classes, feature_names


def _get_categories_from_threshold(feature, threshold, categoric_mappings):
    cat_vals = [f"'{categoric_mappings[feature][int(cat_val)]}'" for cat_val in threshold.split("||")]
    return "["+ ", ".join(cat_vals) + "]"


def _build_python_code_from_tree(function_name, node, feature_names, categoric_mappings):
    code_string = []
    code_string.append(f"def {function_name}({', '.join(feature_names)}):")

    def recurse(node, depth):
        indent = "\t" * depth

        if "leaf_value" not in node:
            feature_name = feature_names[node['split_feature']]
            threshold, default_left, decision_type = node['threshold'], node["default_left"], node["decision_type"]
            if default_left:
                if decision_type == "<=":
                    code_string.append(f"{indent}if {feature_name} <= {threshold} or {feature_name} is None:")
                elif decision_type == "==":
                    selected_categories = _get_categories_from_threshold(feature_name, threshold, categoric_mappings)
                    code_string.append(f"{indent}if {feature_name} in {selected_categories} or {feature_name} is None:")
                elif decision_type == ">=":
                    code_string.append(f"{indent}if {feature_name} > {threshold} or {feature_name} is None:")
                else:
                    raise ValueError(f"Unknown decision_type: {node['decision_type']}")
            else:
                if decision_type == "<=":
                    code_string.append(f"{indent}if {feature_name} <= {threshold}:")
                elif decision_type == "==":
                    selected_categories = _get_categories_from_threshold(feature_name, threshold, categoric_mappings)
                    code_string.append(f"{indent}if {feature_name} in {selected_categories}:")
                elif decision_type == ">=":
                    code_string.append(f"{indent}if {feature_name} > {threshold}:")
                else:
                    raise ValueError(f"Unknown decision_type: {node['decision_type']}")
                
            recurse(node["left_child"], depth + 1)
            code_string.append(f"{indent}else:")
            recurse(node["right_child"], depth + 1)
        else:
            code_string.append(f"{indent}return {node['leaf_value']}")

    recurse(node, 1)
    return "\n".join(code_string)


def get_category_mappings_from_X(X):
    """
    Generates category mappings from the given dataframe X.

    Parameters:
    - X: pandas DataFrame.
        The input dataframe from which category mappings will be generated. Datatype should be "category".

    Returns:
    - categoric_mappings: dict.
        A dictionary containing the category mappings for each categorical column in X.
        The keys of the dictionary are the column names and the values are dictionaries
        representing the category mappings for each column.
    """
    cat_cols = X.select_dtypes("category").columns.tolist()

    def _get_mapper(series):
        return dict(enumerate(series.cat.categories))
    
    if cat_cols:
        categoric_mappings = {col: _get_mapper(X[col]) for col in cat_cols}
    else:
        categoric_mappings = {}

    return categoric_mappings

def export_model(model, categoric_mappings, output_path="lgbm_model_exported.py"):
    """
    Export a trained lightgbm model to a Python code file for inference.

    Args:
        model: The trained model object.
        categoric_mappings: A dictionary containing mappings for categorical features. You may use get_category_mappings_from_X() to generate this dictionary.
        output_path: The path to save the exported Python code file (default is "lgbm_model_exported.py").

    Returns:
        None

    Example:
        X, y = get_data(...) # Get data.
        model = LGBMClassifier(verbose=-1)
        model.fit(X, y)

        categoric_mappings = lgbm_exporter.get_category_mappings_from_X(X)
        lgbm_exporter.export_model(model, categoric_mappings, output_path="lgbm_model_exported.py")


    """
    trees, categoric_mappings, n_classes, feature_names = _get_attributes(model, categoric_mappings)

    # Build n_estimators * n_classes tree codes.
    built_trees = []
    for i, tree in enumerate(trees):
        build_tree = _build_python_code_from_tree(f"tree{i}", tree, feature_names=feature_names, categoric_mappings=categoric_mappings)
        built_trees.append(build_tree)

    # math.exp() vs np.exp() has differences after 12 decimals. To be consistent with LGBM predictions use scipy expit/softmax. 
    import_code = "import numpy as np\nfrom scipy.special import expit, softmax"
    build_trees_code = "\n".join(built_trees)

    # Build final score function
    variable_inputs_as_string = ', '.join(feature_names)
    predict_function_code =  f"def predict({', '.join(feature_names)}):"
    ### For multiclass use softmax
    if n_classes >= 3:
        for n in range(n_classes):
            # Sum all trees' raw scores within each class.
            predict_function_code += f"\n\tfinal_raw_score_class_{n} = " + " + ".join( [f"tree{i}({variable_inputs_as_string})" for i in range(n, n+len(trees), n_classes)])

        predict_function_code += "\n\tfinal_raw_score_array = [" \
                + ",".join( [f'final_raw_score_class_{n}' for n in range(n_classes) ] ) \
                + "]"
        
        predict_function_code += f"\n\treturn softmax(final_raw_score_array)"
    # For binary classification use sigmoid
    elif n_classes == 2:
        predict_function_code += "\n\tfinal_raw_score = " + " + ".join([f"tree{i}({variable_inputs_as_string})" for i in range(len(trees))])
        predict_function_code += f"\n\tfinal_score = expit(final_raw_score)"
        predict_function_code += "\n\treturn np.array([1-final_score, final_score])"
    # For regression final score is the raw score
    elif n_classes == -1: 
        predict_function_code += "\n\tfinal_score = " + " + ".join([f"tree{i}({variable_inputs_as_string})" for i in range(len(trees))])
        predict_function_code += "\n\treturn np.array([final_score])"

    final_code = f"""{import_code}\n\n{build_trees_code}\n\n{predict_function_code}"""

    with open(output_path, "w") as file:
        file.write(final_code)
