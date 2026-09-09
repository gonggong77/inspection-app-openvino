"""
convert_to_ir.py  ─  .keras 모델을 OpenVINO IR(.xml/.bin)로 변환하는 1회성 스크립트
════════════════════════════════════════════════════════════════════
배포용 앱(app.py, infer_keras.py)은 tensorflow 없이 OpenVINO IR만으로 동작한다.
이 스크립트는 그 IR 파일을 만들기 위해 로컬에서 한 번만 실행하면 된다.

실행 방법:
    python convert_to_ir.py
"""

import numpy as np
import tensorflow as tf
import openvino as ov

KERAS_MODEL_PATH = "./weights/leather_model.keras"
IR_MODEL_PATH    = "./weights/leather_model.xml"

model = tf.keras.models.load_model(KERAS_MODEL_PATH)
example_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
ov_model = ov.convert_model(model, example_input=example_input)
ov.save_model(ov_model, IR_MODEL_PATH)
print(f"변환 완료 → {IR_MODEL_PATH}")
