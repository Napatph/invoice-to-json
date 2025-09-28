import streamlit as st
import json
import base64
from openai import OpenAI

# ================= CONFIG =================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
MODEL = "gpt-5"   # change if you have gpt-5-mini or gpt-5-turbo

SYSTEM_PROMPT = """
You are an AI assistant that converts OCR invoice text into structured JSON.

Rules:
1. Always output valid JSON only.
2. Follow this schema exactly:

JSON schema:
{
  "Receiver": "",
  "Tax_ID": "",
  "Branch_No": "00000",
  "Address_EN": "",
  "Address_TH": "",
  "Withholding_Tax_Type_EN": "PND53",
  "Withholding_Tax_Type_TH": "ภงด53",
  "Withholding_Tax_Number": "WT-2025-Temp",
  "Referred_Document_No": "EXP-2025-Temp",
  "Voucher_Number": "PV-2025-Temp",
  "Tax_Condition": "Withheld at source",
  "Invoices": [
    {
      "Invoice_No": "INV-Temp",
      "Date": "",
      "Description": "",
      "Withholding_Tax_Rate": "x%",
      "Pre_VAT_Amount": 0.00,
      "WHT": 0.00
    }
  ],
  "Summary": {
    "Total_Pre_VAT_Amount": 0.00,
    "Total_WHT": 0.00
  }
}

Rules:
- Receiver = company name in English. If no English, then Thai. From which company(Sender), not To/เรียน company. Receiver is just a JSON data field (if available from OCR, else "")
- Tax_ID = tax identification number if found, else ""
- Branch_No = always "00000"
- Address_EN = company address in English (if found)
- Address_TH = company address in Thai (if found)
- Withholding_Tax_Type_EN = always "PND53"
- Withholding_Tax_Type_TH = always "ภงด53"
- Withholding_Tax_Number = always "WT-2025-Temp"
- Referred_Document_No = always "EXP-2025-Temp"
- Voucher_Number = always "PV-2025-Temp"
- Tax_Condition = always "Withheld at source"

Invoices:
- "Invoices" must always be an array of objects in this exact format:
  "Invoices": [
    {
      "Invoice_No": "INV-Temp",
      "Date": "DD/MM/YYYY",
      "Description": "",
      "Withholding_Tax_Rate": "x%",
      "Pre_VAT_Amount": "0.00",
      "WHT": "0.00"
    }
  ]

Invoices Object Rules:
1. One object inside "Invoices" represents **exactly one invoice**.
   - If an invoice has multiple service line items, sum them together into one Pre_VAT_Amount.
   - Do not output individual line items.
2. If multiple invoices are detected in the OCR text, output multiple objects inside the "Invoices" array.
   Example:
   "Invoices": [
     {...Invoice 1...},
     {...Invoice 2...}
   ]
3. "Invoice_No" must always be "INV-Temp"
4. "Date" must always be formatted as DD/MM/YYYY in the Common Era (Gregorian) calendar.
   - If OCR shows Thai Buddhist Era (พ.ศ.), subtract 543 to convert to CE year.
   - Example: "10 กรกฎาคม 2568" → "10/07/2025"
5. "Description" = service description if available, else "".
6. "Withholding_Tax_Rate" = value from keywords like "ภาษีหัก ณ ที่จ่าย x%" or "Withholding tax x%".
7. "Pre_VAT_Amount" = actual value from keywords like "ยอดรวม" or "ยอดรวมหลังหักส่วนลด".
   - Default to 0.00 as number, if missing.
   - Do not put comma and "" for string
8. "WHT" = actual value from same row as "ภาษีหัก ณ ที่จ่าย" or "Withholding tax".
   - Default to 0.00 as number, if missing.
   - Do not put comma and "" for string

Summary:
- Total_Pre_VAT_Amount = 0.00 (calculated later in Python)
- Total_WHT = 0.00 (calculated later in Python)

3. If data is missing, use defaults. Do not invent information.
4. Output must be valid JSON.

Important:
- Output ONLY valid JSON.
- Do not include explanations, comments, or extra text.
- JSON must strictly follow the schema format.

Example of expected output:
{
  "Receiver": "ABC Company (Sender Company)",
  "Tax_ID": "0123456789xxx",
  "Branch_No": "00000",
  "Address_EN": "123/456 ABC Bangkok Thailand",
  "Address_TH": "123/456 ABC กรุงเทพ ประเทศไทย",
  "Withholding_Tax_Type_EN": "PND53",
  "Withholding_Tax_Type_TH": "ภงด53",
  "Withholding_Tax_Number": "WT-2025-Temp",
  "Referred_Document_No": "EXP-2025-Temp",
  "Voucher_Number": "PV-2025-Temp",
  "Tax_Condition": "Withheld at source",
  "Invoices": [
    {
      "Invoice_No": "INV-Temp",
      "Date": "01/07/2025",
      "Description": "Service ABC",
      "Withholding_Tax_Rate": "3%",
      "Pre_VAT_Amount": 50000,
      "WHT": 1500
    },
    {
      "Invoice_No": "INV-Temp",
      "Date": "10/07/2025",
      "Description": "Home charger installation for GAC - 06/2025",
      "Withholding_Tax_Rate": "3%",
      "Pre_VAT_Amount": 60000,
      "WHT": 1800
    },
    {
      "Invoice_No": "INV-Temp",
      "Date": "10/07/2025",
      "Description": "Home charger installation for GAC - 05/2025",
      "Withholding_Tax_Rate": "3%",
      "Pre_VAT_Amount": 80000,
      "WHT": 2400
    }
  ],
  "Summary": {
    "Total_Pre_VAT_Amount": 0.00,
    "Total_WHT": 0.00
  }
}
"""


# ================= HELPERS =================
def encode_image_to_base64(file):
    return base64.b64encode(file.read()).decode("utf-8")


def calculate_summary(data: dict) -> dict:
    total_pre_vat = 0.0
    total_wht = 0.0

    for inv in data.get("Invoices", []):
        try:
            pre_vat = float(str(inv.get("Pre_VAT_Amount", "0")).replace(",", ""))
            inv["Pre_VAT_Amount"] = round(pre_vat, 2)
            total_pre_vat += pre_vat
        except ValueError:
            inv["Pre_VAT_Amount"] = 0.0

        try:
            wht = float(str(inv.get("WHT", "0")).replace(",", ""))
            inv["WHT"] = round(wht, 2)
            total_wht += wht
        except ValueError:
            inv["WHT"] = 0.0

    data["Summary"]["Total_Pre_VAT_Amount"] = round(total_pre_vat, 2)
    data["Summary"]["Total_WHT"] = round(total_wht, 2)
    return data


def process_one_company(uploaded_files):
    client = OpenAI(api_key=OPENAI_API_KEY)

    content = [{"type": "text", "text": "Extract structured JSON from these invoices."}]
    for file in uploaded_files:
        b64 = encode_image_to_base64(file)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    st.write(f"Waiting For GPT to Response")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]
    )



    if hasattr(response, "usage") and response.usage:
        print(f"Tokens used - prompt: {response.usage.prompt_tokens}, "
            f"completion: {response.usage.completion_tokens}, "
            f"total: {response.usage.total_tokens}")
        st.write(f"Tokens used - prompt: {response.usage.prompt_tokens}, "
            f"completion: {response.usage.completion_tokens}, "
            f"total: {response.usage.total_tokens}")

    raw_output = response.choices[0].message.content
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        st.error("❌ GPT did not return valid JSON.")
        st.text(raw_output)
        raise

    return calculate_summary(data)


def process_multiple_companies(invoice_sets):
    companies = []
    for idx, files in enumerate(invoice_sets, start=1):
        st.write(f"📌 Processing company {idx} with {len(files)} invoices...")
        company_data = process_one_company(files)
        companies.append(company_data)
    return {"companies": companies}


# ================= STREAMLIT UI =================
st.set_page_config(page_title="Invoice to JSON", layout="wide")

st.title("📑 Invoice → JSON Extractor")
st.write("Upload invoice files for each company. You can add or delete companies.")

if "invoice_sets" not in st.session_state:
    st.session_state.invoice_sets = [[]]  # start with one company

# Button to add another company (place it above or below the loop, both work with rerun)
if st.button("➕ Add another company"):
    st.session_state.invoice_sets.append([])
    st.rerun()

# Uploaders for each company
to_delete = None  # track which company to delete
for i, invoice_files in enumerate(st.session_state.invoice_sets):
    st.subheader(f"Company {i+1}")

    cols = st.columns([3, 1])
    with cols[0]:
        uploaded = st.file_uploader(
            f"Upload invoices for company {i+1}",
            type=["png", "jpg", "jpeg", "pdf"],
            accept_multiple_files=True,
            key=f"uploader_{i}"
        )
        if uploaded:
            st.session_state.invoice_sets[i] = uploaded

    with cols[1]:
        if st.button("🗑️ Delete", key=f"delete_{i}"):
            to_delete = i

# Actually delete after loop
if to_delete is not None:
    del st.session_state.invoice_sets[to_delete]
    if not st.session_state.invoice_sets:
        st.session_state.invoice_sets = [[]]  # keep at least one block
    st.rerun()

# Process button
if st.button("🚀 Process All Companies"):
    if any(st.session_state.invoice_sets):
        result = process_multiple_companies(st.session_state.invoice_sets)

        st.subheader("✅ Final Output JSON")
        st.json(result)

        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        st.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name="all_companies.json",
            mime="application/json"
        )
    else:
        st.warning("Please upload at least one invoice.")
