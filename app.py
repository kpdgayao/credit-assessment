import streamlit as st
from supabase import create_client
import os
from dotenv import load_dotenv
import uuid
import io
import requests
import PyPDF2
import json
import anthropic
from weasyprint import HTML, CSS  # Add this import statement at the top of your script
import logging

# Load environment variables from .env file
load_dotenv()

# Set up Anthropic API client
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_api_url = "https://api.anthropic.com/v1/messages"

# Set up Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_api_key = os.getenv("SUPABASE_API_KEY")
supabase_client = create_client(supabase_url, supabase_api_key)

@st.cache_data(show_spinner=False)  # Cache extraction results
def extract_data_from_pdf(file):
    try:
        # Set a file size limit (e.g., 5MB)
        max_size = 5 * 1024 * 1024
        if file.size > max_size:
            st.warning("File size exceeds the limit of 5MB. Please upload a smaller file.")
            return None

        # Create a BytesIO object from the uploaded file
        file_bytes = io.BytesIO(file.read())

        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file_bytes)

        # Extract text from each page of the PDF
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        return text
    except Exception as e:
        st.error(f"Error occurred during text extraction: {str(e)}")
        return None

@st.cache_data(show_spinner=False)  # Cache processed data
def process_extracted_data(extracted_data):
    try:
        # Use Anthropic API to process the extracted data
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        client = anthropic.Client(api_key=anthropic_api_key)

        messages = [
            {"role": "user", "content": f"Please process the following extracted data from a PDF loan application form:\n\n{extracted_data}\n\nProvide the processed data in a structured JSON format."}
        ]

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=messages,
            max_tokens=1024
        )

        processed_data = response.content[0].text

        # Attempt to parse the processed data as JSON
        try:
            processed_data = json.loads(processed_data)
        except json.JSONDecodeError:
            # If parsing fails, return the processed data as a string
            processed_data = {"data": processed_data}

        return processed_data

    except anthropic.BadRequestError as e:
        st.error(f"Bad Request Error occurred while processing the extracted data. Error details: {str(e)}")
        return None
    except anthropic.APIError as e:
        st.error(f"API Error occurred while processing the extracted data. Error details: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while processing the extracted data. Error details: {str(e)}")
        return None

def store_processed_data(processed_data):
    try:
        # Store the processed data in Supabase
        processed_data_record = {
            "id": str(uuid.uuid4()),
            "data": processed_data
        }
        response = supabase_client.table("processed_data").insert(processed_data_record).execute()
        if response:
            st.success("Processed data stored successfully.")
        else:
            st.error("Failed to store processed data.")
    except Exception as e:
        st.error(f"Error occurred while storing processed data: {str(e)}")

@st.cache_data(show_spinner=False)  # Cache generated credit report
def generate_credit_report(processed_data):
    try:
        # Use Anthropic API to generate credit assessment report
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        client = anthropic.Client(api_key=anthropic_api_key)

        system_prompt = """
You are an AI-powered credit assessment expert for individual borrowers in the Philippines. Your task is to thoroughly analyze the provided loan application details and generate a comprehensive credit report with a rigorous initial assessment of the applicant's creditworthiness based on the 5 Cs of credit: Capacity, Capital, Character, Collateral, and Conditions.

For each of the 5 Cs, provide an in-depth analysis and a conservative score on a scale of 1 to 5, with 5 being the strongest. Be strict in your evaluation and identify any missing information, inconsistencies, or red flags that require further investigation or clarification. Do not make any assumptions in favor of the applicant.

Based on your analysis, provide an overall credit score (1-5) and a recommendation on whether to approve, conditionally approve, or deny the loan application. If you conditionally approve or deny the application, provide specific suggestions on what additional information, documents, or changes to the loan terms could potentially lead you to reconsider your decision.

In your report, cite specific information from the loan application to support your analysis. Highlight any additional documents, such as detailed credit reports, verified financial statements, or professional collateral appraisals, that would be required to make a fully informed assessment.

Format your report in Markdown as follows:

# Credit Assessment Report
**Application ID:** [ID]
**Processing Date:** [YYYY-MM-DD HH:MM:SS]
**Prepared For:** [Initials]

## Loan Details
**Loan Amount:** [Amount]
**Loan Term:** [Term]

## 5 C's Analysis
### Capacity - Score: [1-5]
[Analysis]

### Capital - Score: [1-5]
[Analysis]

### Character - Score: [1-5]
[Analysis]

### Collateral - Score: [1-5]
[Analysis]

### Conditions - Score: [1-5]
[Analysis]

## Overall Credit Score
**[Score] out of 5**

## Recommendation
**[Approve/Conditionally Approve/Conditionally Deny/Deny]**
[Explanation]

## Suggestions for Reconsideration
[Suggestions, if applicable]

## Risk Explanation
*[High/Medium/Low risk, with explanation]*

Do not include any personally identifiable information in the report. Instead, refer to the applicant using only their initials, and reference the application ID, processing date and time, and the initials of the person who requested the report.

Your report should be clear, well-structured, and objective, with a focus on identifying and mitigating potential risks. Avoid making any biased or discriminatory assessments. The goal is to provide a rigorous, data-driven initial assessment to guide further underwriting while carefully considering all factors that could impact the borrower's likelihood of repaying the loan as agreed.

Remember, as a prudent credit assessor, your primary responsibility is to protect the lender's interests and maintain a high-quality loan portfolio. Do not hesitate to request additional information or recommend denial if the application does not meet strict underwriting standards.

However, keep in mind that your assessment is a suggestion to the credit committee, not a final decision. The credit committee will review your report along with other relevant information to make the ultimate determination on loan approval.

Please include the following notices at the end of your report:

---
**Disclaimer:**
*This AI-generated credit assessment is provided as a tool to assist the credit committee in making an informed decision. It should not be relied upon as the sole basis for loan approval or denial. The credit committee should independently verify the information provided, request additional documentation as needed, and carefully consider all relevant factors before making a final determination on the loan application.*

**Confidentiality Notice:**
*This credit assessment report is confidential and intended solely for the use of the credit committee and authorized underwriters. It should not be shared, reproduced, or distributed to any third parties without express written permission. This report is part of an experimental AI-based credit evaluation system and should be used judiciously in conjunction with established underwriting practices.*
"""

        messages = [
            {"role": "user", "content": f"Loan Application Details:\n{json.dumps(processed_data)}\n\nPlease generate the credit assessment report."}
        ]

        try:
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                system=system_prompt,
                messages=messages
            )
        except anthropic.APIError as e:
            st.error(f"API Error occurred while generating the credit report. Error details: {str(e)}")
            # Log the error details for further investigation
            logging.error(f"Anthropic API Error: {str(e)}")
            return None

        report = response.content[0].text

        return report
    except anthropic.BadRequestError as e:
        st.error(f"Bad Request Error occurred while generating the credit report. Error details: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while generating the credit report. Error details: {str(e)}")
        return None

def store_credit_report(credit_report):
    try:
        # Store the credit report in Supabase
        credit_report_record = {
            "id": str(uuid.uuid4()),
            "report": credit_report
        }
        response = supabase_client.table("credit_reports").insert(credit_report_record).execute()
        if response:
            st.success("Credit report stored successfully.")
        else:
            st.error("Failed to store credit report.")
    except Exception as e:
        st.error(f"Error occurred while storing credit report: {str(e)}")

def html_to_pdf(report):
    try:
        css_file = "styles.css"  # Path to your CSS file

        # Wrap the report in HTML tags and add classes for formatting
        formatted_report = f"""
        <html>
        <head>
            <style>
                /* Add any custom styles here */
            </style>
        </head>
        <body>
            {report}
        </body>
        </html>
        """

        html = HTML(string=formatted_report)
        css = CSS(filename=css_file)
        return html.write_pdf(stylesheets=[css])
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

def main():
    st.title("OCCC Credit Assessment Report")  # Rename the title
    
    with st.expander("Upload Loan Application Form"):
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        
    if uploaded_file:
        with st.spinner("Extracting data..."):
            extracted_data = extract_data_from_pdf(uploaded_file)
            
        if extracted_data:
            with st.spinner("Analyzing data..."):
                processed_data = process_extracted_data(extracted_data)
                
            if processed_data:
                with st.spinner("Generating report..."):
                    credit_report = generate_credit_report(processed_data)
                    
                    # Display Report in Expandable Section
                    with st.expander("View Credit Assessment Report"):
                        st.markdown(credit_report)
                        
                if credit_report:
                    with st.spinner("Storing report..."):
                        store_credit_report(credit_report)
                        
                    with st.spinner("Generating PDF..."):
                        try:
                            pdf_bytes = html_to_pdf(credit_report)
                            st.download_button(
                                label="Download PDF",
                                data=pdf_bytes,
                                file_name="occc_credit_assessment_report.pdf",
                                mime="application/pdf",
                            )
                        except Exception as e:
                            st.error(f"Error generating PDF: {e}")
                else:
                    st.error("Failed to generate credit report.")
            else:
                st.error("Failed to process data.")
        else:
            st.error("Failed to extract data.")

if __name__ == "__main__":
    main()