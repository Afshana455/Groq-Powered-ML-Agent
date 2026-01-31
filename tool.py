from langchain.tools import StructuredTool
from model_tool import predict_target

predict_tool = StructuredTool.from_function(
    func=predict_target,
    name="ML_Predictor",
    description="Predict target value from a list of 8 numeric features"
)
