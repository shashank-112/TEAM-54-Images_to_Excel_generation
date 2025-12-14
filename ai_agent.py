import json
import torch
from ultralytics import YOLO

import os
from llm_brain import LLMBrain


import T1_PdfToImageConverterFile as pic
import T2_TableBitsRemoverFile as tbr
import T3_ImageToNumberExtractor as ine
import T4_NumberToExcelSheet as nes


class AgentState:
    def __init__(self):
        self.pdf_loaded = False
        self.images_extracted = False
        self.marks_extracted = False
        self.excel_created = False


class DocumentProcessingAgent:
    def __init__(
        self,
        data_path,
        yolo_model_path,
        cnn_model_path,
        output_excel_name,
        no_of_absentees,
        year,
        branch,
        section,
        student_data_path
    ):
        self.goal = "Convert PDF to structured Excel"
        self.state = AgentState()
        api_key = os.getenv("OPENAI_API_KEY")

        if api_key:
            print("🤖 LLM enabled (OpenAI)")
            self.llm = LLMBrain(api_key)
        else:
            print("🤖 LLM disabled (no API key). Using rule-based commands.")
            self.llm = None


        self.data_path = data_path
        self.output_excel_name = output_excel_name
        self.no_of_absentees = no_of_absentees
        self.year = year
        self.branch = branch
        self.section = section

        # Load student data
        with open(student_data_path, "r") as f:
            self.student_data = json.load(f)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load YOLO
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.model.to(self.device)

        # Load CNN
        self.cnn_model = pic.StrongCNN()
        self.cnn_model.load_state_dict(
            torch.load(cnn_model_path, map_location=self.device)
        )
        self.cnn_model.to(self.device)
        self.cnn_model.eval()

        self.all_images = None
        self.all_marks = []
        self.last_pg_no = 0

    def chat(self):
        print("\n🤖 AI Agent is online.")
        print("Type 'help' to see available commands.\n")

        while True:
            user_input = input("🧑 You: ").strip()

            # 1️⃣ Get command
            if self.llm:
                command = self.llm.interpret(user_input)
            else:
                command = user_input

            # 2️⃣ Normalize command
            command = command.strip().lower().replace("'", "").replace('"', "")

            print(f"🤖 LLM decided command: {command}")

            # 3️⃣ Act on command
            if command in ["help", "show_help"]:
                print("""
    🤖 Available Commands:
    - start / run / process pdf
    - status
    - goal
    - exit / quit
    """)

            elif command in ["run_agent", "start", "run", "process pdf"]:
                print("🤖 Agent: Starting document processing...\n")
                self.run()

            elif command in ["show_status", "status"]:
                print(f"""
    🤖 Agent Status:
    - PDF Loaded        : {self.state.pdf_loaded}
    - Images Extracted : {self.state.images_extracted}
    - Marks Extracted  : {self.state.marks_extracted}
    - Excel Created    : {self.state.excel_created}
    """)

            elif command in ["show_goal", "goal"]:
                print(f"🤖 Agent Goal: {self.goal}")

            elif command in ["exit", "quit"]:
                print("🤖 Agent: Shutting down. Goodbye!")
                break

            else:
                print("🤖 Agent: I did not understand that command. Type 'help'.")


    
    
    
    
    
    
    

    # ------------------ AGENT PERCEPTION ------------------
    def observe(self):
        if self.data_path.endswith(".pdf"):
            self.state.pdf_loaded = True


    # ------------------ AGENT DECISION ------------------
    def decide(self):
        if not self.state.images_extracted:
            return "EXTRACT_IMAGES"

        if not self.state.marks_extracted:
            return "EXTRACT_MARKS"

        if not self.state.excel_created:
            return "CREATE_EXCEL"

        return "DONE"


    # ------------------ AGENT ACTION ------------------
    def act(self, action):
        if action == "EXTRACT_IMAGES":
            image_extractor = pic.PdfToImageConverter(self.data_path)
            self.all_images, self.no_of_images = image_extractor.execute()
            self.state.images_extracted = True

        elif action == "EXTRACT_MARKS":
            for pg_no, image in enumerate(self.all_images):
                box_extractor = tbr.SmallBoxDetector(pg_no, image, self.yolo_model)
                single_image_34_boxes = box_extractor.execute()

                digit_extractor = ine.ImageToDigitExtractor(
                    single_image_34_boxes,
                    pg_no,
                    self.cnn_model,
                    self.device
                )
                marks = digit_extractor.execute()
                self.all_marks.append(marks.copy())

                self.last_pg_no = pg_no   # ✅ save safely

            self.state.marks_extracted = True


        elif action == "CREATE_EXCEL":
            xl_maker = nes.ArangeDataInExcel(
                self.all_marks,
              self.output_excel_name,
                self.last_pg_no,
                self.no_of_absentees,
                student_details=self.student_data[self.year][self.branch][self.section]
            )
            xl_maker.execute()
            self.state.excel_created = True



    # ------------------ AGENT LOOP ------------------
    def run(self):
        print(f"🤖 AI Agent started | Goal: {self.goal}")

        self.observe()

        while True:
            action = self.decide()
            print(f"🤖 Agent action: {action}")

            if action == "DONE":
                print("✅ Goal achieved. Agent stopped.")
                break

            self.act(action)
