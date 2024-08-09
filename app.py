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
logging.basicConfig(level=logging.ERROR)
from xhtml2pdf import pisa
from tenacity import retry, stop_after_attempt, wait_exponential
import re

st.set_page_config(page_title="Credit Assessor")

# Load environment variables from .env file
load_dotenv()

@st.cache_resource
def get_anthropic_client():
    return anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

@st.cache_resource
def get_supabase_client():
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_API_KEY"))

# Set up Anthropic API client
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Set up Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_api_key = os.getenv("SUPABASE_API_KEY")
supabase_client = get_supabase_client()

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
        client = get_anthropic_client()

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

def extract_application_id(credit_report):
    match = re.search(r'<p><strong>Application ID:</strong>\s*(.*?)\s*</p>', credit_report)
    if match:
        return match.group(1).strip()
    return "unknown"

@st.cache_data(show_spinner=False)  # Cache generated credit report
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_credit_report(processed_data):
    try:
        # Use Anthropic API to generate credit assessment report
        client = get_anthropic_client()

        system_prompt = """
        You are an AI-powered credit assessment expert for individual borrowers in the Philippines. Your task is to thoroughly analyze the provided loan application details and generate a comprehensive credit report with a rigorous initial assessment of the applicant's creditworthiness based on the 5 Cs of credit: Capacity, Capital, Character, Collateral, and Conditions.

        For each of the 5 Cs, provide an in-depth analysis and a conservative score on a scale of 1 to 5, with 5 being the strongest. Be strict in your evaluation and identify any missing information, inconsistencies, or red flags that require further investigation or clarification. Do not make any assumptions in favor of the applicant.

        Based on your analysis, provide an overall credit score (1-5) and a recommendation on whether to approve, conditionally approve, or deny the loan application. If you conditionally approve or deny the application, provide specific suggestions on what additional information, documents, or changes to the loan terms could potentially lead you to reconsider your decision.

        In your report, cite specific information from the loan application to support your analysis. Highlight any additional documents, such as detailed credit reports, verified financial statements, or professional collateral appraisals, that would be required to make a fully informed assessment.

        Format your report in HTML with the following structure:

        <html>
        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    font-size: 12px;
                    line-height: 1.5;
                }
                h1, h2, h3, h4, h5, h6 {
                    margin-top: 20px;
                    margin-bottom: 10px;
                }
                h1 {
                    font-size: 24px;
                }
                h2 {
                    font-size: 20px;
                }
                h3 {
                    font-size: 16px;
                }
                p {
                    margin-bottom: 10px;
                }
                strong {
                    font-weight: bold;
                }
                em {
                    font-style: italic;
                }
                ul, ol {
                    margin-left: 20px;
                    margin-bottom: 10px;
                }
                li {
                    margin-bottom: 5px;
                }
            </style>
        </head>
        <body>
            <h1>Credit Assessment Report</h1>
            <p><strong>Application ID:</strong> [ID]</p>
            <p><strong>Processing Date:</strong> [YYYY-MM-DD HH:MM:SS]</p>
            <p><strong>Prepared For:</strong> [Initials]</p>

            <h2>Loan Details</h2>
            <p><strong>Loan Amount:</strong> [Amount]</p>
            <p><strong>Loan Term:</strong> [Term]</p>

            <h2>5 C's Analysis</h2>
            <h3>Capacity - Score: [1-5]</h3>
            <p>[Analysis]</p>

            <h3>Capital - Score: [1-5]</h3>
            <p>[Analysis]</p>

            <h3>Character - Score: [1-5]</h3>
            <p>[Analysis]</p>

            <h3>Collateral - Score: [1-5]</h3>
            <p>[Analysis]</p>

            <h3>Conditions - Score: [1-5]</h3>
            <p>[Analysis]</p>

            <h2>Overall Credit Score</h2>
            <p><strong>[Score] out of 5</strong></p>

            <h2>Recommendation</h2>
            <p><strong>[Approve/Conditionally Approve/Conditionally Deny/Deny]</strong></p>
            <p>[Explanation]</p>

            <h2>Suggestions for Reconsideration</h2>
            <p>[Suggestions, if applicable]</p>

            <h2>Risk Explanation</h2>
            <p><em>[High/Medium/Low risk, with explanation]</em></p>

            <hr>
            <p><strong>Disclaimer:</strong></p>
            <p><em>This AI-generated credit assessment is provided as a tool to assist the credit committee in making an informed decision. It should not be relied upon as the sole basis for loan approval or denial. The credit committee should independently verify the information provided, request additional documentation as needed, and carefully consider all relevant factors before making a final determination on the loan application.</em></p>

            <p><strong>Confidentiality Notice:</strong></p>
            <p><em>This credit assessment report is confidential and intended solely for the use of the credit committee and authorized underwriters. It should not be shared, reproduced, or distributed to any third parties without express written permission. This report is part of an experimental AI-based credit evaluation system and should be used judiciously in conjunction with established underwriting practices.</em></p>
        </body>
        </html>

        Do not include any personally identifiable information in the report. Instead, refer to the applicant using only their initials, and reference the application ID, processing date and time, and the initials of the person who requested the report.

        Your report should be clear, well-structured, and objective, with a focus on identifying and mitigating potential risks. Avoid making any biased or discriminatory assessments. The goal is to provide a rigorous, data-driven initial assessment to guide further underwriting while carefully considering all factors that could impact the borrower's likelihood of repaying the loan as agreed.

        Remember, as a prudent credit assessor, your primary responsibility is to protect the lender's interests and maintain a high-quality loan portfolio. Do not hesitate to request additional information or recommend denial if the application does not meet strict underwriting standards.

        However, keep in mind that your assessment is a suggestion to the credit committee, not a final decision. The credit committee will review your report along with other relevant information to make the ultimate determination on loan approval.
        """

        messages = [
            {"role": "user", "content": f"Loan Application Details:\n{json.dumps(processed_data)}\n\nPlease generate the credit assessment report."}
        ]

        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=3000,
                system=system_prompt,
                messages=messages
            )
        except anthropic.APIError as e:
            logging.error(f"Anthropic API Error: {str(e)}")
            st.error(f"API Error occurred while generating the credit report. Error details: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error in generate_credit_report: {str(e)}")
            st.error(f"An unexpected error occurred while generating the credit report. Error details: {str(e)}")
            return None

        report = response.content[0].text
       
        # Wrap the report in a <div> tag with a specific class
        formatted_report = f'<div class="credit-report">{report}</div>'
        
        return formatted_report
    
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
            st.success("Credit report saved successfully.")
        else:
            st.error("Failed to process credit report.")
    except Exception as e:
        st.error(f"Error occurred while processing credit report: {str(e)}")

def html_to_pdf(report):
    try:
        # Add inline CSS for PDF styling
        styled_report = f"""
        <style>
            body {{
                font-family: Arial, sans-serif;
                font-size: 12px;
                line-height: 1.5;
                color: black;
                background-color: white;
            }}
            h1, h2, h3, h4, h5, h6 {{
                margin-top: 20px;
                margin-bottom: 10px;
                color: black;
            }}
            p {{
                margin-bottom: 10px;
            }}
            ul, ol {{
                margin-left: 20px;
                margin-bottom: 10px;
            }}
            li {{
                margin-bottom: 5px;
            }}
        </style>
        {report}
        """
        html = HTML(string=styled_report)
        css = CSS(string='@page { size: A4; margin: 1cm }')
        return html.write_pdf(stylesheets=[css])
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

def extract_application_id(credit_report):
    # Extract the application ID from the credit report using regular expressions or string manipulation
    # Modify this function based on the structure of your generated report
    # Example implementation:
    start_index = credit_report.find("Application ID:") + len("Application ID:")
    end_index = credit_report.find("</p>", start_index)
    if start_index != -1 and end_index != -1:
        application_id = credit_report[start_index:end_index].strip()
        return application_id
    else:
        return "application_id_not_found"

def main():
    st.title("OCCC Credit Assessment Report")
    
    with st.expander("Upload Loan Application Form"):
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        
    if uploaded_file:
        progress_bar = st.progress(0)
        
        with st.spinner("Extracting data..."):
            extracted_data = extract_data_from_pdf(uploaded_file)
            progress_bar.progress(25)
            
        if extracted_data:
            with st.spinner("Analyzing data..."):
                processed_data = process_extracted_data(extracted_data)
                progress_bar.progress(50)
                
            if processed_data:
                with st.spinner("Generating report..."):
                    credit_report = generate_credit_report(processed_data)
                    progress_bar.progress(75)
                    
                    # Display Report in Expandable Section
                    with st.expander("View Credit Assessment Report"):
                        # Add CSS for light and dark mode compatibility
                        st.markdown("""
                            <style>
                            .credit-report {
                                font-family: Arial, sans-serif;
                                font-size: 14px;
                                line-height: 1.5;
                                color: var(--text-color);
                                background-color: var(--background-color);
                                padding: 20px;
                                border-radius: 5px;
                            }
                            .credit-report h1, .credit-report h2, .credit-report h3, 
                            .credit-report h4, .credit-report h5, .credit-report h6 {
                                color: var(--text-color);
                                margin-top: 20px;
                                margin-bottom: 10px;
                            }
                            .credit-report p {
                                margin-bottom: 10px;
                            }
                            .credit-report ul, .credit-report ol {
                                margin-left: 20px;
                                margin-bottom: 10px;
                            }
                            .credit-report li {
                                margin-bottom: 5px;
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        
                        # Wrap the credit report in a div with the credit-report class
                        styled_report = f'<div class="credit-report">{credit_report}</div>'
                        
                        # Display the styled report
                        st.components.v1.html(styled_report, height=600, scrolling=True)

                progress_bar.progress(100)

                if credit_report:
                    with st.spinner("Storing report..."):
                        store_credit_report(credit_report)
                        
                    with st.spinner("Generating PDF..."):
                        try:
                            pdf_bytes = html_to_pdf(credit_report)
                            
                            if pdf_bytes:
                                application_id = extract_application_id(credit_report)
                                file_name = f"{application_id}_credit_assessment_report.pdf"
                                
                                st.download_button(
                                    label="Download PDF",
                                    data=pdf_bytes,
                                    file_name=file_name,
                                    mime="application/pdf",
                                )
                                st.success("Credit report processed successfully. You can now download the PDF.")
                            else:
                                st.error("Failed to generate PDF.")
                        except Exception as e:
                            st.error(f"Error generating PDF: {e}")
                else:
                    st.error("Failed to generate credit report.")
            else:
                st.error("Failed to process data.")
        else:
            st.error("Failed to extract data.")
    else:
        st.info("Please upload a loan application form in PDF format.")

if __name__ == "__main__":
    main()