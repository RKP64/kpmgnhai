import streamlit as st
import fitz  # PyMuPDF for annotation extraction and text reading
import pandas as pd
import io
from fpdf import FPDF

# LLM & Vector Store imports (for the AI Q&A layer)
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq

#########################################
# STREAMLIT CONFIG & STYLES
#########################################
st.set_page_config(page_title="KPMG Digital Assistant", layout="wide")
st.markdown(
    """
    <style>
    body { background-color: #f7f9fc; }
    .stButton>button {
        background-color: #4caf50; color: white;
        border: none; border-radius: 5px;
        padding: 8px 15px; font-size: 14px;
        margin: 5px; cursor: pointer;
    }
    .stButton>button:hover { background-color: #45a049; }
    .response-box {
        background-color: #ffffff; border-radius: 10px;
        padding: 20px; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        font-family: Arial, sans-serif; font-size: 16px;
        color: #333333; line-height: 1.6; margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#########################################
# LOAD API KEYS FROM STREAMLIT SECRETS
#########################################
groq_api_key = st.secrets["groq_api_key"]
openai_api_key = st.secrets["openai_api_key"]

#########################################
# SESSION STATE INITIALIZATION
#########################################
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "latest_annots_df" not in st.session_state:
    st.session_state.latest_annots_df = pd.DataFrame()
if "ai_annotation_response" not in st.session_state:
    st.session_state.ai_annotation_response = ""

#########################################
# SIDEBAR SETTINGS
#########################################
st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox(
    "Select Model:",
    [
        "llama-3.3-70b-versatile",
        "deepseek-r1-distill-llama-70b",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
max_context_length = st.sidebar.number_input("Max Context Length (tokens):", 1000, 8000, 3000)
retrieve_mode = st.sidebar.selectbox("Retrieve Mode:", ["Text (Hybrid)", "Vector Only", "Text Only"])
# Toggle for document chunking (vector store creation)
enable_chunking = st.sidebar.checkbox(
    "Enable other AI features", value=True
)

#########################################
# FILE UPLOAD
#########################################
st.header("EDA Assistant - Annotation Extraction & AI Q&A")
uploaded_files = st.file_uploader("Upload PDF(s):", type=["pdf"], accept_multiple_files=True)

#########################################
# HELPER: Extract All Annotations Using PyMuPDF
#########################################
def extract_all_annotations_in_range(file_bytes, start_page, end_page):
    """
    Opens the PDF from file_bytes using PyMuPDF and extracts all annotations
    (of any type) from pages [start_page, end_page). It also attempts to parse
    a clause from the annotation content (if present).
    Returns a list of dictionaries.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    annotations = []
    total_pages = len(doc)
    if end_page > total_pages:
        end_page = total_pages

    import re
    clause_pattern = re.compile(r"Clause\s*([\d\.]+):", re.IGNORECASE)

    for page_idx in range(start_page, end_page):
        page = doc[page_idx]
        annots = page.annots()
        if not annots:
            continue
        for annot in annots:
            info = annot.info
            content = info.get("content", "").strip()
            clause_match = clause_pattern.search(content)
            clause = clause_match.group(1) if clause_match else "null"
            annotations.append({
                "Page No.": page_idx + 1,
                "Type": annot.type[0],
                "Clause": clause,
                "Content": content,
                "Author": info.get("title", "Unknown"),
                "Subject": info.get("subject", ""),
                "CreationDate": info.get("creationDate", ""),
                "ModDate": info.get("modDate", "")
            })
    return annotations

#########################################
# (OPTIONAL) HELPER: Chunk PDF Text for Document Q&A (RAG)
#########################################
def chunk_pdf_text(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = []
    for page in doc:
        full_text.append(page.get_text())
    combined_text = "\n".join(full_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(combined_text)
    return chunks

#########################################
# PROCESS UPLOADED FILES FOR DOCUMENT Q&A (Optional)
#########################################
vector_store = None
if uploaded_files and enable_chunking:
    st.subheader("Processing Documents for Q&A")
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.type == "application/pdf":
                file_bytes = uploaded_file.read()
                pdf_chunks = chunk_pdf_text(file_bytes)
                embeddings = OpenAIEmbeddings(api_key=openai_api_key)
                if vector_store is None:
                    vector_store = FAISS.from_texts(pdf_chunks, embeddings)
                else:
                    temp_vs = FAISS.from_texts(pdf_chunks, embeddings)
                    vector_store.merge_from(temp_vs)
                st.success(f"Processed (and chunked): {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

#########################################
# PREBUILT PAGE-RANGE PROMPTS FOR ANNOTATION EXTRACTION
#########################################
page_prompts = [
    "Extract annotations from pages 0–50",
    "Extract annotations from pages 51–100",
    "Extract annotations from pages 101–150",
    "Extract annotations from pages 151–200",
    "Extract annotations from pages 201–250",
]

st.subheader("Step 1: Extract Raw Annotations")
selected_prompt = st.radio("Select a page-range prompt or type your own:", page_prompts)
custom_prompt = st.text_input("Or type your custom page-range request:")

#########################################
# SUBMIT: Raw Annotation Extraction
#########################################
if st.button("Extract Raw Annotations"):
    final_prompt = custom_prompt if custom_prompt else selected_prompt
    start_page, end_page = 0, 0
    if selected_prompt == "Extract annotations from pages 0–50":
        start_page, end_page = 0, 51
    elif selected_prompt == "Extract annotations from pages 51–100":
        start_page, end_page = 51, 101
    elif selected_prompt == "Extract annotations from pages 101–150":
        start_page, end_page = 101, 151
    elif selected_prompt == "Extract annotations from pages 151–200":
        start_page, end_page = 151, 201
    elif selected_prompt == "Extract annotations from pages 201–250":
        start_page, end_page = 201, 251

    if uploaded_files:
        pdf_file = next((f for f in uploaded_files if f.type == "application/pdf"), None)
        if pdf_file:
            file_bytes = pdf_file.getvalue()
            annots = extract_all_annotations_in_range(file_bytes, start_page, end_page)
            if annots:
                df_annots = pd.DataFrame(annots)
                st.subheader("Raw Annotation Table")
                st.dataframe(df_annots, use_container_width=True)
                st.session_state.latest_annots_df = df_annots.copy()
                st.session_state.conversation_history.append({
                    "question": final_prompt,
                    "response": f"Extracted {len(annots)} annotations from pages {start_page} to {end_page-1}."
                })
            else:
                st.info("No annotations found in the selected page range.")
        else:
            st.warning("No PDF file found among uploads.")
    else:
        st.warning("Please upload a document first.")

#########################################
# STEP 2: AI Q&A on Annotations (Annotation Summary)
#########################################
st.subheader("Step 2: AI Q&A on Annotations")
ai_question = st.text_input("Enter your AI question/instruction regarding the annotations:",
                              "Summarize and group the annotations by common issues.")
if st.button("Generate AI Annotation Response"):
    if st.session_state.latest_annots_df.empty:
        st.warning("Please extract annotations first.")
    else:
        # Build a bullet list from the raw annotations
        bullet_points = []
        for _, row in st.session_state.latest_annots_df.iterrows():
            bullet_points.append(
                f"- Page {row['Page No.']}, Clause: {row['Clause']}, Content: {row['Content']}, Author: {row['Author']}"
            )
        combined_annots = "\n".join(bullet_points)
        final_ai_prompt = f"{ai_question}\n\nAnnotations:\n{combined_annots}"
        system_message = {
            "role": "system",
            "content": (
                "You are an AI assistant specialized in analyzing PDF annotations. "
                "Based on the provided annotations, answer the question concisely and group similar issues together."
            )
        }
        user_message = {
            "role": "user",
            "content": final_ai_prompt
        }
        try:
            llm = ChatGroq(model_name=selected_model, api_key=groq_api_key)
            response = llm.invoke([system_message, user_message], temperature=temperature)
            ai_output = response.content.strip()
            st.subheader("AI Annotation Response")
            st.markdown(f"<div class='response-box'>{ai_output}</div>", unsafe_allow_html=True)
            st.session_state.ai_annotation_response = ai_output
            st.session_state.conversation_history.append({
                "question": "AI Q&A on annotations: " + ai_question,
                "response": ai_output
            })
        except Exception as e:
            st.error(f"Error generating AI annotation response: {str(e)}")

#########################################
# STEP 3: Document Q&A (General Q&A using RAG)
#########################################
st.subheader("Step 3: Document Intelligence")
doc_question = st.text_input("Enter your question about the document:", "What are the main topics discussed?")
if st.button("Generate Document Q&A Response"):
    if vector_store is None:
        st.warning("Document vector store is not built. To enable Document Q&A, please check the chunking option in the sidebar and re-upload your document.")
    else:
        docs = vector_store.similarity_search(doc_question, k=5)
        context = " ".join([d.page_content for d in docs])
        if len(context) > max_context_length:
            context = context[:max_context_length]
        system_message = {
            "role": "system",
            "content": (
                "You are an AI assistant that answers questions based solely on the provided document context. "
                "Answer the question concisely and factually."
            )
        }
        user_message = {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {doc_question}"
        }
        try:
            llm = ChatGroq(model_name=selected_model, api_key=groq_api_key)
            response = llm.invoke([system_message, user_message], temperature=temperature)
            doc_qa_output = response.content.strip()
            st.subheader("Document Q&A Response")
            st.markdown(f"<div class='response-box'>{doc_qa_output}</div>", unsafe_allow_html=True)
            st.session_state.conversation_history.append({
                "question": "Document Q&A: " + doc_question,
                "response": doc_qa_output
            })
        except Exception as e:
            st.error(f"Error generating document Q&A response: {str(e)}")

#########################################
# DOWNLOAD BUTTONS: PDF & EXCEL
#########################################
def generate_pdf(conversation_history):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Assistant Response", ln=True, align='C')
    pdf.ln(10)
    if conversation_history:
        last_entry = conversation_history[-1]
        q = last_entry.get("question", "").encode('latin-1', 'replace').decode('latin-1')
        r = last_entry.get("response", "").encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, f"Q: {q}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"A: {r}")
        pdf.ln(10)
    output = pdf.output(dest="S")  # Get the output from FPDF
    if isinstance(output, str):
        # If it's a string (from very old fpdf), encode it to latin-1
        output_bytes = output.encode('latin-1', 'replace')
    else:
        # If it's already bytes (from fpdf2), use it directly
        output_bytes = output
    return io.BytesIO(output_bytes)

def generate_excel_from_df(df: pd.DataFrame):
    if df.empty:
        return None
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Annotations")
    excel_buffer.seek(0)
    return excel_buffer

if st.session_state.conversation_history:
    st.download_button(
        label="Download Conversation as PDF",
        data=generate_pdf(st.session_state.conversation_history),
        file_name="conversation.pdf",
        mime="application/pdf"
    )

if not st.session_state.latest_annots_df.empty:
    excel_data = generate_excel_from_df(st.session_state.latest_annots_df)
    if excel_data:
        st.download_button(
            label="Download Raw Annotations as Excel",
            data=excel_data,
            file_name="raw_annotations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
