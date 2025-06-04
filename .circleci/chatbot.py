import logging
import os
from pathlib import Path
import re
import time
import pandas as pd
from playwright.sync_api import Playwright, sync_playwright
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("temp/mixer_par_npar_definition_search.log"),
        logging.StreamHandler()
    ]
)

def extract_par_mixer_definition_details(playwright: Playwright,
                                         claim_type: str,
                                         service_from_date: str,
                                         mixer_fetch_number: None,
                                         par_fetch_number: None) -> None:
    browser = playwright.chromium.launch(channel="chrome", headless=False)
    context = browser.new_context(http_credentials={
        "username": os.getenv("PEGA_USERNAME"),
        "password": os.getenv("PEGA_PASSWORD"),
    },
    accept_downloads=True,
    )
    page = context.new_page()
    page.goto("https://pmc.uat.antheminc.com/prweb/PRAuth/SSO")

    # Wait for either login page or landing page
    page.wait_for_load_state("networkidle")

    current_url = page.url

    if re.match(r"https://secure-fed\.uat\.anthem\.com/as/.*/resume/as/authorization\.ping", current_url):
        # Not logged in, perform login
        page.locator("#username").fill(os.getenv("PEGA_USERNAME"))
        page.locator("#password").fill(os.getenv("PEGA_PASSWORD"))
        page.get_by_text("Submit").click()
        # Optionally wait for landing page after login
        page.wait_for_load_state()
    else:
        logging.info("Already logged in, skipping login steps.")

    # page.wait_for_url(re.compile(r"https://secure-fed\.uat\.anthem\.com/as/.*/resume/as/authorization\.ping"))
    
    # Move mouse to the left edge so that the navbar appears
    page.mouse.move(10, 300)
    page.get_by_role("menuitem", name=" Mixer").click()
    page.wait_for_load_state("networkidle")

    # Wait for "Mixer" button text
    mixer_text = page.locator(
        "iframe[name=\"PegaGadget0Ifr\"]"
    ).content_frame.locator(
        "[data-test-id=\"\\32 02007270744300208593_header\"]"
    ).get_by_role("button").inner_text(timeout=10000)
    if "Mixer" not in mixer_text:
        raise Exception("Mixer button not found or text mismatch.")

    # Wait for "Mixer Data" button text
    mixer_data_text = page.locator(
        "iframe[name=\"PegaGadget0Ifr\"]"
    ).content_frame.locator(
        "[data-test-id=\"\\32 02105071050000776403_header\"]"
    ).get_by_role("button").inner_text(timeout=10000)
    if "Mixer Data" not in mixer_data_text:
        raise Exception("Mixer Data button not found or text mismatch.")

    # Wait for "Intraplan Search" button text
    intraplan_text = page.locator(
        "iframe[name=\"PegaGadget0Ifr\"]"
    ).content_frame.locator(
        "[data-test-id=\"\\32 02007300842570961532_header\"]"
    ).get_by_role("button").inner_text(timeout=10000)
    if "Intraplan Search" not in intraplan_text:
        raise Exception("Intraplan Search button not found or text mismatch.")
    
    # Select "Claim Type" - Professional (P) or Institutional (I) or Both (B)
    page.locator("iframe[name=\"PegaGadget0Ifr\"]").content_frame.locator("[data-test-id=\"\\32 02007270744300209206208\"]").select_option(claim_type)

    # Select "Service From Date" - 5/11/2025
    page.locator("iframe[name=\"PegaGadget0Ifr\"]").content_frame.locator("[data-test-id=\"\\32 02007270744300212213698\"]").fill(service_from_date)
    
    # Select "Service Thru Date" - 12/31/9999
    page.locator("iframe[name=\"PegaGadget0Ifr\"]").content_frame.locator("[data-test-id=\"\\32 02007270744300213214304\"]").fill("12/31/9999")
    
    # Click "Search" button
    page.locator("iframe[name=\"PegaGadget0Ifr\"]").content_frame.locator("[data-test-id=\"\\32 01911280422460287138707\"]").click()

    page.wait_for_load_state("networkidle")

    # Wait for the search results to load
    results_text = page.locator(
        "iframe[name=\"PegaGadget0Ifr\"]"
    ).content_frame.locator(
        "[data-test-id=\"\\32 020050306042607747849\"]"
    ).inner_text()

    if not re.search(r"\d+ results found", results_text):
        raise Exception("No results found.")
    
    # mixer_df = pd.DataFrame(columns=[, 
    #                                  "Network/Contract", "Pricing Contract", "Pricing Variance", "Par Ind"])

    # mixer_df["Business Type"] = pd.Series(dtype="string")
    # mixer_df["Mixer Ind F"] = pd.Series(dtype="string")
    # mixer_df["Mixer Ind W"] = pd.Series(dtype="string")
    

    # total_search_results = int(re.search(r"(\d+) results found", results_text).group(1))

    if mixer_fetch_number is None:
        number_of_pages = int(page.locator(
            "iframe[name=\"PegaGadget0Ifr\"]"
            ).content_frame.locator(
                "[data-test-id=\"\\32 019112808553400251\"] tr"
                ).filter(
                    has_text="Export to ExcelExport Non Par"
                    ).locator(
                        "[data-test-id=\"\\32 0141121165713061615380\"]"
                        ).inner_text())
    else:
        number_of_pages = mixer_fetch_number
    j = 0

    while number_of_pages > 0:
        i = 1
        page_input = page.locator(
            "iframe[name=\"PegaGadget0Ifr\"]"
            ).content_frame.locator('input[name="pyGridActivePage"]').first
        # Get the aria-label attribute
        aria_label = page_input.get_attribute("aria-label")
        # Extract the first number using regex
        current_page = int(re.search(r"\b(\d+)\b", aria_label).group(1))

        while i <= 10:
            j = j + 1
            logging.info(f"Processing overall row {j} and current row {i} on page {current_page}")
            try:
                row_locator = page.locator(
                    "iframe[name=\"PegaGadget0Ifr\"]"
                    ).content_frame.locator(
                        f"[data-test-id=\"\\32 019112808553400251-R{j}\"] [data-test-id=\"\\32 02007221237320152105630\"]"
                        )

                if not row_locator.is_visible(timeout=3000):
                    logging.info(f"Row {j} is not visible. Breaking inner loop.")
                    break

                row_locator.click()

                page.locator(
                    "iframe[name=\"PegaGadget0Ifr\"]"
                    ).content_frame.get_by_role("menuitem", name="View Mixer Details").click()

                
                business_type = page.locator(
                    "iframe[name=\"PegaGadget1Ifr\"]"
                    ).content_frame.locator("div").filter(
                        has_text=re.compile(r"^Business typeno value$")
                        ).locator("[data-test-id=\"\\32 020031808065500477439\"]").inner_text()

                mixer_ind_f = page.locator(
                    "iframe[name=\"PegaGadget1Ifr\"]"
                    ).content_frame.locator("div").filter(
                        has_text=re.compile(r"^MIX IND F Return value to CLMno value$")
                        ).locator("[data-test-id=\"\\32 020031808065500477439\"]").inner_text()

                mixer_ind_w = page.locator(
                    "iframe[name=\"PegaGadget1Ifr\"]"
                    ).content_frame.locator("div").filter(
                        has_text=re.compile(r"^MIX IND W Return value to CLMno value$")
                        ).locator("[data-test-id=\"\\32 020031808065500477439\"]").inner_text()

                # logging.info(f"Business Type: {business_type}")
                # mixer_df.at[j, "Business Type"] = business_type
                # logging.info(f"Mixer Ind F: {mixer_ind_f}")
                # mixer_df.at[j, "Mixer Ind F"] = mixer_ind_f
                # logging.info(f"Mixer Ind W: {mixer_ind_w}")
                # mixer_df.at[j, "Mixer Ind W"] = mixer_ind_w

                code_spans = page.locator(
                                    "iframe[name=\"PegaGadget1Ifr\"]"
                                ).content_frame.locator(
                                    "span[data-test-id=\"202007291253110027323989\"]"
                                ).all_inner_texts()

                # Assign by index (adjust indices if order is different)
                last_alphabet = (code_spans[1] if len(code_spans) > 1 else "").split()[0][0]
                product_code = code_spans[2] if len(code_spans) > 2 else ""
                coverage_code = code_spans[3] if len(code_spans) > 3 else ""
                variance = code_spans[4] if len(code_spans) > 4 else ""
                i = i + 1
                logging.info(f"Tab Name: {product_code}{coverage_code}{variance}P{last_alphabet}")

                # EXTRACT PAR RESULTS
                # Wait for the iframe to be available
                iframe_selector = f'iframe[title="{product_code}{coverage_code}{variance}P{last_alphabet}"]'
                page.wait_for_selector(iframe_selector, timeout=10000)

                # Use FrameLocator to find the divs
                results_locator = page.frame_locator(iframe_selector).locator("div.dataLabelRead")

                par_results = None
                for text in results_locator.all_inner_texts():
                    logging.info(f"dataLabelRead text: {text}")
                    if re.search(r"\d+\s+results found", text):
                        par_results = text
                        break

                if par_results is None:
                    logging.warning("No PAR results div found.")
                else:
                    logging.info(f"PAR Results: {par_results}")

                number_of_results = int(re.search(r"(\d+) results found", par_results).group(1))
                if number_of_results > 20:
                    try:
                        number_of_par_pages = int(
                            page.locator(f"iframe[title=\"{product_code}{coverage_code}{variance}P{last_alphabet}\"]"
                                        ).content_frame.locator(
                                            "div.dataLabelRead"
                                            ).filter(
                                                has_text=re.compile(r"^\d+$")
                                                ).first.inner_text())
                    except Exception as e:
                        logging.exception(f"Error occurred: {e}")
                        number_of_par_pages = 1
                elif number_of_results == 0:
                    number_of_par_pages = 0
                else:
                    number_of_par_pages = 1
                
                if par_fetch_number is not None and number_of_par_pages != 0:
                    number_of_par_pages = par_fetch_number

                logging.info(f"Number of PAR results: {number_of_results} -- Number of pages: {number_of_par_pages}")

                par_df = pd.DataFrame(columns=["Business Type", "Mixer Ind F", "Mixer Ind W",
                                               "Product Code", "Coverage Code", 
                                                "Variance", "Priority Order",
                                                "Network/Contract", "Pricing Contract",
                                                "Pricing Variance", "Par Ind"])

                while number_of_par_pages > 0:
                    if number_of_results > 20:
                        current_page = int(page.locator(
                            f"iframe[title=\"{product_code}{coverage_code}{variance}P{last_alphabet}\"]"
                            ).content_frame.get_by_role(
                                "textbox", name="Page 1 of").input_value())
                    else:
                        current_page = 1
                    
                    content_frame = page.locator(f"iframe[title=\"{product_code}{coverage_code}{variance}P{last_alphabet}\"]").content_frame

                    # Extract headers
                    header_cells = content_frame.locator("table.gridTable > tbody > tr").first.locator("th")
                    headers = [header_cells.nth(i).locator(".cellIn").inner_text().strip() for i in range(header_cells.count())]
                    headers = ["Business Type", "Mixer Ind F", "Mixer Ind W", "Product Code", "Coverage Code", "Variance"] + headers
                    logging.info(f"Par Table Headers: {headers}")

                    try:
                        table_locator = content_frame.locator('table#bodyTbl_right').nth(0)
                        row_locators = table_locator.locator("tbody > tr.oddRow, tbody > tr.evenRow")
                        logging.info(f"Extracting {row_locators.count()} rows on PAR page {current_page}")
                        for row_idx in range(row_locators.count()):
                            rows = [business_type, mixer_ind_f, mixer_ind_w, product_code, coverage_code, variance]
                            cell_locators = row_locators.nth(row_idx).locator("td")
                            row_data = [cell_locators.nth(i).locator("span").inner_text().strip() for i in range(cell_locators.count())]
                            rows.extend(row_data)
                            logging.info(f"Par Table Rows - {row_idx}: {rows}")
                            temp_par_df = pd.DataFrame([rows], columns=headers)
                            par_df = pd.concat([par_df, temp_par_df], ignore_index=True)

                    except Exception as e:
                        logging.exception(f"Error occurred: {e}")
                        break
                        
                    next_button_locator = page.locator(f"iframe[title=\"{product_code}{coverage_code}{variance}P{last_alphabet}\"]").content_frame.get_by_role("button", name=">", exact=True)
                    
                    if next_button_locator.count() > 0 and next_button_locator.is_visible(timeout=1000):
                        next_button_locator.click(timeout=3000)
                        page.wait_for_load_state("networkidle")
                    else:
                        logging.info("Next button not found or not visible, ending PAR page loop.")
                    
                    number_of_par_pages = number_of_par_pages - 1
                    logging.info(f"Number of PAR pages left: {number_of_par_pages}")
                
                try:
                    saved_par_df = pd.read_excel(Path(__file__).parent.parent / "temp/par_results.xlsx")
                    par_df = pd.concat([saved_par_df, par_df], ignore_index=True)
                    par_df.to_excel(Path(__file__).parent.parent / "temp/par_results.xlsx", index=False)
                except FileNotFoundError:
                    par_df.to_excel(Path(__file__).parent.parent / "temp/par_results.xlsx", index=False)
                logging.info("PAR results saved to par_results.xlsx")

            except Exception as e:
                logging.error(f"Error occurred: {e}")
                break

            finally:
                try:
                    page.get_by_role(
                        "tab", name=f"{product_code}{coverage_code}{variance}P{last_alphabet} Press Delete to"
                    ).get_by_label("Close this tab").click(timeout=3000)
                except Exception as close_e:
                    logging.warning(f"Could not close tab: {close_e}")
        
        # get the next page
        next_page_locator = page.locator(
            "iframe[name=\"PegaGadget0Ifr\"]"
            ).content_frame.locator(
                "[data-test-id=\"\\32 019112808553400251\"] tr"
                ).filter(
                    has_text="Export to ExcelExport Non Par"
                    ).locator(
                        "button[name=\"pyGridPaginator_pyDisplayHarness\\.CPRActions_9\"]"
                        )
        if not next_page_locator.is_visible(timeout=2000):
            logging.info("Next page locator is not visible. Breaking inner loop.")
            break

        next_page_locator.click()
        page.wait_for_load_state("networkidle")

        page_input = page.locator(
            "iframe[name=\"PegaGadget0Ifr\"]"
            ).content_frame.locator('input[name="pyGridActivePage"]').first
        # Get the aria-label attribute
        aria_label = page_input.get_attribute("aria-label")
        # Extract the first number using regex
        current_page = int(re.search(r"\b(\d+)\b", aria_label).group(1))

        number_of_pages =  number_of_pages - 1
        logging.info(f"Number of Mxer details pages left: {number_of_pages}")

   
    # mixer_df.to_excel(Path(__file__).parent.parent / "temp/mixer_search_results.xlsx",
    #     sheet_name="Mixer",
    #     index=False
    # )
    # ---------------------
    page.close()
    context.close()
    browser.close()


with sync_playwright() as playwright:
    start = time.time()
    logging.info("Starting Mixer Par Definition extraction")
    try:
      extract_par_mixer_definition_details(playwright,
                                           "P",
                                           "1/1/2025",
                                           mixer_fetch_number=1,
                                           par_fetch_number=1)
    except Exception as e:
      logging.error(f"Error occurred during extraction: {e}")
    finally:
      end = time.time()
      logging.info(f"Extraction completed in {end - start:.2f} seconds")
      logging.info("--------------------------------------------------")
