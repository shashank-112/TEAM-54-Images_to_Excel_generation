flowchart LR

%% =========================
%% Main Stream (Horizontal)
%% =========================

M1[A. PDF Upload]
M2[B. PDF to Images]
M3[C. YOLO Box Coordinate Extraction]
M4[D. Crop Bounding Boxes]
M5[E. Image Transformations]
M6[F. ResNet-based CNN Digit Prediction]
M7[G. Excel Sheet Creation]

M1 --> M2 --> M3 --> M4 --> M5 --> M6 --> M7


%% ==================================
%% Side Stream - A (Right to Left)
%% YOLO Training Pipeline
%% ==================================

SA1[A. Box Annotation using MakeSense.ai]
SA2[B. Image Augmentation]
SA3[C. YOLO Model Training]
SA4[D. Model Testing]
SA5[E. Model Deployment]

SA5 --> SA4 --> SA3 --> SA2 --> SA1
SA1 --> M3


%% ==================================
%% Side Stream - B (Left to Right)
%% Digit Recognition Training Pipeline
%% ==================================

SB1[A. Digit Data Collection]
SB2[B. Create Digit Classes 0 to 9 and Blank]
SB3[C. Add Noise to Data]
SB4[D. Convert Data to CSV Format]
SB5[E. Train ResNet-based CNN]
SB6[F. Deploy Digit Recognition Model]

SB1 --> SB2 --> SB3 --> SB4 --> SB5 --> SB6
SB6 --> M6
