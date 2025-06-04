# Debug: Print all visible buttons in the pagination area
pagination_area = page.locator(
    "iframe[name=\"PegaGadget0Ifr\"]"
).content_frame.locator(
    "[data-test-id=\"\\32 019112808553400251\"] tr"
).filter(
    has_text="Export to ExcelExport Non Par"
)

# Log what buttons are available
buttons = pagination_area.locator("button").all()
for i, btn in enumerate(buttons):
    try:
        logging.info(f"Button {i}: text='{btn.inner_text()}', name='{btn.get_attribute('name')}'")
    except:
        pass
