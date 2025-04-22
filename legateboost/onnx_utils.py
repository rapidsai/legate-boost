from typing import Any, List

import numpy as np

import cupynumeric as cn

# onnx is imported only if needed - keep this a soft dependency
try:
    import onnx
except ImportError:
    pass


def make_model(graph: Any) -> Any:
    # make model with appropriate opset imports for legate-boost
    LEGATEBOOST_ONNX_OPSET_IMPORTS = [
        onnx.helper.make_opsetid("ai.onnx.ml", 3),
        onnx.helper.make_opsetid("", 21),
    ]
    return onnx.helper.make_model(graph, opset_imports=LEGATEBOOST_ONNX_OPSET_IMPORTS)


def reshape_predictions(graph: Any, pred: cn.ndarray) -> Any:
    # àppend an onnx graph that shapes the predictions equivalently to pred
    shape = list(pred.shape)
    shape[0] = -1
    out_type = "int64" if pred.dtype == cn.int64 else "double"
    onnx_text = f"""
    ReshapePredictions ({out_type}[N, M] predictions_in) => ({out_type}{shape} predictions_out)
    {{
        shape = Constant<value_ints={shape}>()
        predictions_out = Reshape(predictions_in, shape)
    }}
    """  # noqa: E501
    reshape_graph = onnx.parser.parse_graph(onnx_text)
    graph = onnx.compose.merge_graphs(
        graph,
        reshape_graph,
        io_map=[
            (graph.output[0].name, "predictions_in"),
        ],
        prefix2="reshape_",
    )
    return graph


def mirror_predict_proba_output(graph: Any) -> Any:
    # where model outputs only true probability we need to add the false probability
    onnx_text = """
    MirrorPredict (double[N, M] predictions_in) => (double[N, 2] predictions_out)
    {
        one = Constant<value = double {1.0}>()
        false_probability = Sub(one, predictions_in)
        predictions_out = Concat<axis=1>(false_probability, predictions_in)
    }
    """  # noqa: E501
    new_graph = onnx.parser.parse_graph(onnx_text)
    new_graph = onnx.compose.merge_graphs(
        graph,
        new_graph,
        io_map=[
            (graph.output[0].name, "predictions_in"),
        ],
        prefix2="mirror_",
    )
    return new_graph


def init_predictions(model_init: cn.array, X_dtype: Any) -> Any:
    # form a graph that takes X_in and model_init as input and outputs
    # model_init repeated n_rows times

    X_type_text = "double" if X_dtype == cn.float64 else "float"
    onnx_text = f"""
    InitPredictions ({X_type_text}[N, M] X_in) => ({X_type_text}[N, M] X_out, double[N, K] predictions_out)
    {{
        X_out = Identity(X_in)
        n_rows = Shape<end=1>(X_in)
        one = Constant<value_ints=[1]>()
        tile_repeat = Concat<axis=0>(n_rows, one)
        predictions_out = Tile(init, tile_repeat)
    }}
    """  # noqa: E501
    graph = onnx.parser.parse_graph(onnx_text)
    graph.initializer.append(
        onnx.numpy_helper.from_array(np.atleast_2d(model_init.__array__()), name="init")
    )
    return graph


def merge_model_graphs(graphs: List[Any]) -> Any:
    # merge a list of graphs into a single graph
    combined = graphs[0]
    for i, g in enumerate(graphs[1:]):
        combined = onnx.compose.merge_graphs(
            combined,
            g,
            io_map=[
                (combined.output[0].name, "X_in"),
                (combined.output[1].name, "predictions_in"),
            ],
            prefix2="model_{}_".format(i),
        )
    return combined
