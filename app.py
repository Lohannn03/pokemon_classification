import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps

MODEL_PATH = "best_model.pth"
CLASS_PATH = "class_names.txt"
SAMPLE_DIR = "sample_images"
IMG_SIZE = 224

device = torch.device("cpu")

st.set_page_config(
    page_title="Pokemon Classifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #202020;
        color: white;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: white;
    }

    .title-box {
        background-color: #d90429;
        color: #ffd60a;
        padding: 22px 28px;
        border-radius: 4px;
        font-size: 34px;
        font-weight: 900;
        margin-bottom: 16px;
    }

    .desc-text {
        color: #dddddd;
        font-size: 14px;
        margin-bottom: 10px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #252525;
        border: 2px solid #666666;
        border-radius: 4px;
        min-height: 430px;
    }

    [data-testid="stFileUploader"] {
        background-color: #2b2b2b;
        border-radius: 4px;
        padding: 8px;
    }

    [data-testid="stFileUploader"] section {
        padding: 8px;
        min-height: 70px;
    }

    [data-testid="stFileUploader"] small {
        font-size: 11px;
    }

    div.stButton > button {
        width: 100%;
        height: 30px;
        background-color: #333333;
        color: white;
        border: 1px solid #777777;
        border-radius: 4px;
        font-size: 12px;
        padding: 2px 6px;
    }

    div.stButton > button:hover {
        background-color: #d90429;
        color: white;
        border: 1px solid #d90429;
    }

    .result-title {
        text-align: center;
        color: white;
        font-size: 26px;
        font-weight: 800;
        margin-top: 6px;
        margin-bottom: 12px;
    }

    .confidence {
        text-align: center;
        color: #ffd60a;
        font-size: 18px;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .top5 {
        font-size: 13px;
        color: #eeeeee;
        line-height: 1.55;
    }

    .sample-caption {
        font-size: 10px;
        color: #cccccc;
        overflow-wrap: break-word;
        margin-top: -8px;
        margin-bottom: 8px;
    }

    .gallery-scroll {
        max-height: 520px;
        overflow-y: auto;
        padding-right: 4px;
    }

    .small-note {
        color: #bbbbbb;
        font-size: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, len(class_names))
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    return model, class_names


model, classes = load_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def get_sample_images():
    if not os.path.exists(SAMPLE_DIR):
        return []

    image_paths = []

    for file_name in os.listdir(SAMPLE_DIR):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_paths.append(os.path.join(SAMPLE_DIR, file_name))

    return image_paths


def make_preview(image, size=(170, 120)):
    preview = image.copy().convert("RGB")
    preview = ImageOps.fit(preview, size, method=Image.Resampling.LANCZOS)
    return preview


def make_display_image(image, size=(260, 220)):
    display = image.copy().convert("RGB")
    display.thumbnail(size, Image.Resampling.LANCZOS)
    return display


def predict_image(image):
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        top_probs, top_idxs = torch.topk(probs, k=5, dim=1)

    return top_probs[0], top_idxs[0]


st.markdown(
    "<div class='title-box'>Pokemon Classifier</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='desc-text'>Upload an image or select a sample image from the right gallery to classify the Pokemon.</div>",
    unsafe_allow_html=True
)

left_col, result_col, gallery_col = st.columns([1.15, 1.15, 0.95], gap="medium")

selected_image = None
selected_name = None

sample_paths = get_sample_images()

with gallery_col:
    st.subheader("Sample Images")

    if len(sample_paths) == 0:
        st.warning("No sample images found. Please add images to the sample_images folder.")
    else:
        st.markdown("<div class='gallery-scroll'>", unsafe_allow_html=True)

        for row_start in range(0, len(sample_paths), 2):
            cols = st.columns(2, gap="small")

            for i in range(2):
                idx = row_start + i

                if idx >= len(sample_paths):
                    break

                img_path = sample_paths[idx]
                img_name = os.path.basename(img_path)

                with cols[i]:
                    img = Image.open(img_path).convert("RGB")
                    preview = make_preview(img)

                    st.image(preview, use_container_width=True)

                    if st.button("Select", key=f"select_{img_path}"):
                        st.session_state["selected_sample"] = img_path

                    short_name = img_name
                    if len(short_name) > 22:
                        short_name = short_name[:22] + "..."

                    st.markdown(
                        f"<div class='sample-caption'>{short_name}</div>",
                        unsafe_allow_html=True
                    )

        st.markdown("</div>", unsafe_allow_html=True)

with left_col:
    with st.container(border=True):
        st.subheader("Input Image")

        uploaded_file = st.file_uploader(
            "Drag and drop an image here, or click to select a file",
            type=["jpg", "jpeg", "png", "webp"]
        )

        st.markdown("<div class='small-note'>or</div>", unsafe_allow_html=True)

        st.text_input(
            "Paste image URL here",
            placeholder="Paste image URL here",
            disabled=True,
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file).convert("RGB")
            selected_name = uploaded_file.name

        elif "selected_sample" in st.session_state:
            selected_path = st.session_state["selected_sample"]
            selected_image = Image.open(selected_path).convert("RGB")
            selected_name = os.path.basename(selected_path)

        if selected_image is not None:
            input_display = make_display_image(selected_image, size=(280, 220))
            st.image(input_display, caption=selected_name, use_container_width=False)
        else:
            st.info("Upload an image or click Select on a sample image.")

with result_col:
    with st.container(border=True):
        st.subheader("Prediction Result")

        if selected_image is not None:
            top_probs, top_idxs = predict_image(selected_image)

            pred_idx = top_idxs[0].item()
            pred_name = classes[pred_idx]
            pred_conf = top_probs[0].item()

            st.markdown(
                f"<div class='result-title'>It's {pred_name}!</div>",
                unsafe_allow_html=True
            )

            result_display = make_display_image(selected_image, size=(300, 220))
            st.image(result_display, use_container_width=False)

            st.markdown(
                f"<div class='confidence'>Confidence: {pred_conf:.2%}</div>",
                unsafe_allow_html=True
            )

            st.markdown("<div class='top5'><b>Top 5 Predictions</b></div>", unsafe_allow_html=True)

            top5_html = "<div class='top5'>"

            for i in range(5):
                idx = top_idxs[i].item()
                prob = top_probs[i].item()
                top5_html += f"{i + 1}. {classes[idx]} - {prob:.2%}<br>"

            top5_html += "</div>"

            st.markdown(top5_html, unsafe_allow_html=True)

        else:
            st.markdown(
                "<div class='result-title'>Prediction Result</div>",
                unsafe_allow_html=True
            )
            st.write("The predicted Pokemon name and image will appear here.")