# click on 3 dots to view details
    icon_locator = page.locator(
        "iframe[name=\"PegaGadget0Ifr\"]").content_frame.locator("i[data-test-id=\"2020102204111909914861\"]")

    icon_locator.wait_for(state="visible", timeout=10000)
    icon_locator.click()
    
    # click on view details
    page.locator(
        "iframe[name=\"PegaGadget0Ifr\"]").content_frame.locator(
            "[data-test-id=\"\\32 02010220410560385239\"]").get_by_role("menuitem", name="View").click()
    
    view_code_set_results = page.locator("iframe[name=\"PegaGadget1Ifr\"]").content_frame.locator(
        "[data-test-id=\"\\32 020050306042607747849\"]").inner_text(timeout=10000)
    
    logging.info(f"View Code Set Results: {view_code_set_results}")
    

    classification_code_df = pd.DataFrame(columns=["Code Set Value Type", "Classification Code", 
                                                   "Classification Description", "Classification Type Code",
                                                   "Notes", "Code", "Code description", "Code Effective Date",
                                                   "Code Termination Date", "Suspend Effective Date",
                                                   "Suspend End Date", "Relationship Effective Date",
                                                   "Relationship End Date", "Status",])

    # Extract headers
    header_cells = page.locator("iframe[name=\"PegaGadget1Ifr\"]").content_frame.locator("table.gridTable > tbody > tr").first.locator("th")
    headers = [header_cells.nth(i).locator(".cellIn").inner_text().strip() for i in range(header_cells.count())]
    headers = ["Code Set Value Type", "Classification Code", "Classification Description", "Classification Type Code", "Notes"] + headers
    logging.info(f"Par Table Headers: {headers}")

    try:
        code_set_value_type = page.locator(
            "iframe[name=\"PegaGadget1Ifr\"]").content_frame.locator(
                "[data-test-id=\"\\32 02010230916110067109285\"]").inner_text().strip()
        classification_code = page.locator(
            "iframe[name=\"PegaGadget1Ifr\"]").content_frame.locator(
                "[data-test-id=\"\\32 0200318080655004470574\"]").inner_text().strip()
        classification_description = page.locator(
            "iframe[name=\"PegaGadget1Ifr\"]").content_frame.locator(
                "[data-test-id=\"\\32 020031808065500426733\"]").inner_text().strip()
        classification_type_code = page.locator(
            "iframe[name=\"PegaGadget1Ifr\"]").content_frame.locator(
                "[data-test-id=\"\\32 0200318080655004368927\"]").inner_text().strip()
        notes = page.locator(
            "iframe[name=\"PegaGadget1Ifr\"]").content_frame.locator(
                "[data-test-id=\"\\32 02010230613200022118150\"]").inner_text().strip()

        logging.info(f"Code Set Value Type: {code_set_value_type}, Classification Code: {classification_code}, "
                     f"Classification Description: {classification_description}, Classification Type Code: {classification_type_code}, " 
                     f"Notes: {notes}")
    

        table_locator = page.locator("iframe[name=\"PegaGadget1Ifr\"]").content_frame.locator('table#bodyTbl_right').nth(0)
        row_locators = table_locator.locator("tbody > tr.oddRow, tbody > tr.evenRow")

        for row_idx in range(row_locators.count()):
            rows = [code_set_value_type, classification_code, classification_description, classification_type_code, notes]
            cell_locators = row_locators.nth(row_idx).locator("td")
            row_data = []
            for i in range(cell_locators.count()):
                # Prefer visible span with text, fallback to empty string if not found
                cell_span = cell_locators.nth(i).locator("span:visible")
                try:
                    value = cell_span.inner_text().strip()
                except Exception:
                    value = ""
                row_data.append(value)
            rows.extend(row_data)
            logging.info(f"Par Table Rows - {row_idx}: {rows}")
            temp_classification_code_df = pd.DataFrame([rows], columns=headers)
            classification_code_df = pd.concat([classification_code_df, temp_classification_code_df], ignore_index=True)

    except Exception as e:
        logging.exception(f"Error occurred: {e}")
