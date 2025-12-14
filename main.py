from ai_agent import DocumentProcessingAgent

OUTPUT_EXCEL_SHEET_NAME = "output"
DATA_PATH = "./asserts/pdf_or_data/6.pdf"
YOLO_MODEL_PATH = "./asserts/model/best.pt"
CNN_MODEL_PATH = "./asserts/model/final_28x28_model.pt"
NO_OF_ABSENTEES = [[2, 3, 5]]
YEAR = 3
BRANCH = 0
SECTION = 0
STUDENT_DATA_PATH = "./asserts/students_data/data.json"


agent = DocumentProcessingAgent(
    data_path=DATA_PATH,
    yolo_model_path=YOLO_MODEL_PATH,
    cnn_model_path=CNN_MODEL_PATH,
    output_excel_name=OUTPUT_EXCEL_SHEET_NAME,
    no_of_absentees=NO_OF_ABSENTEES,
    year=YEAR,
    branch=BRANCH,
    section=SECTION,
    student_data_path=STUDENT_DATA_PATH
)

agent.chat()

