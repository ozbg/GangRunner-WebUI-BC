# app.py
# 

from flask import Flask, render_template, request
from OriginalGangRunner_Area_Colour_V9 import (
    parse_input_data,
    parse_json_input,
    calculate_a4_units,
    calculate_sheet_capacity,
    find_best_combination_ilp,
    get_configuration,
    Order,
    Sheet
)
import traceback
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_text():
    input_text = request.form.get('input_text', '').strip()
    mode = '1'  # Default to Mode 1

    if not input_text:
        return render_template('index.html', error="Input text is required.", input_text=input_text)

    try:
        # Retrieve sheet parameters from form   --   
        min_sheets = request.form.get('min_sheets', '245').strip()
        max_sheets = request.form.get('max_sheets', '500').strip()
        increment = request.form.get('increment', '5').strip()

        # Validate and convert sheet parameters
        try:
            min_sheets = int(min_sheets)
            max_sheets = int(max_sheets)
            increment = int(increment)
            if min_sheets <= 0 or max_sheets <= 0 or increment <= 0:
                raise ValueError
            if min_sheets > max_sheets:
                return render_template('index.html', error="Minimum sheets cannot be greater than maximum sheets.", input_text=input_text)
        except ValueError:
            return render_template('index.html', error="Invalid sheet parameters. Please enter positive integers with min sheets ≤ max sheets.", input_text=input_text)

        # Determine configuration based on selected mode
        # Since mode selection is removed, always use Mode 1 settings
        high_priority_mode = False
        custom_multiplier = 1
        config = get_configuration(high_priority_mode, custom_multiplier, min_sheets, max_sheets, increment)
        mode_description = "Normal Mode"

        # Parse input data into Order objects and skipped lines
        orders, skipped_lines = parse_input_data(input_text)

        if not orders and not skipped_lines:
            return render_template('index.html', error="No valid data found.", input_text=input_text)

        # Calculate A4 units for each order
        for order in orders:
            calculate_a4_units(order)

        # Define sheet size (e.g., 590mm x 847mm)
        sheet = Sheet(590, 847)
        calculate_sheet_capacity(sheet)

        # Perform ILP Optimization
        # Iterate over possible waste weights and sheet counts
        best_overall_result = None
        max_gp_per_point = float('-inf')  # Initialize max GP per Point

        for waste_weight in config['possible_waste_weights']:
            config['waste_weight'] = waste_weight  # Set the current waste weight

            for sheets_fixed in config['possible_sheets']:
                best_combination = find_best_combination_ilp(orders, sheet, sheets_fixed, config)
                if best_combination:
                    # Calculate Gross Profit for this combination
                    plate_cost_per_run = config['plate_cost_per_run']
                    paper_cost_per_sheet = config['paper_cost_per_sheet']
                    courier_cost_per_order = config['courier_cost_per_order']

                    total_run_value = sum(order.value for order in best_combination['orders'])

                    total_overs_cost = sum(
                        (best_combination['produced_qty'][order.order_id] - order.quantity) * (order.value / order.quantity)
                        for order in best_combination['orders']
                    )

                    total_plate_cost = plate_cost_per_run
                    total_paper_cost = best_combination['sheet_count'] * paper_cost_per_sheet

                    # **Courier Cost Calculation**
                    unique_base_order_ids = set(order.base_order_id for order in best_combination['orders'])
                    total_courier_cost = len(unique_base_order_ids) * courier_cost_per_order

                    # **Calculate Overhead Costs**
                    overhead_cost = 0.09 * total_run_value  # 9% of total run value

                    # **Update Total Production Cost to include Overhead Costs**
                    total_production_cost = total_plate_cost + total_paper_cost + total_courier_cost + overhead_cost  # Excluding overs_cost

                    # **Calculate Gross Profit by excluding overs_cost**
                    gross_profit = total_run_value - total_production_cost - total_overs_cost  # Keep as is after correcting overs_cost

                    # Calculate Points per Run and GP per Point
                    points_per_run = 1500 + best_combination['sheet_count']
                    gp_per_point = gross_profit / points_per_run if points_per_run != 0 else 0

                    # Keep track of the best overall result based on GP per Point
                    if gp_per_point > max_gp_per_point:
                        max_gp_per_point = gp_per_point
                        # Assign Total Unique Kinds and Total Placed Objects separately
                        total_unique_kinds = len(best_combination['orders'])
                        total_placed_objects = sum(best_combination['ups_per_order'].values())
                        best_overall_result = {
                            'best_combination': best_combination,
                            'gross_profit': gross_profit,
                            'waste_weight': waste_weight,
                            'points_per_run': points_per_run,
                            'gp_per_point': gp_per_point,
                            'total_unique_kinds': total_unique_kinds,
                            'total_placed_objects': total_placed_objects,
                            'total_run_value': total_run_value,
                            'total_overs_cost': total_overs_cost,
                            'total_courier_cost': total_courier_cost,
                            'total_plate_cost': total_plate_cost,
                            'total_paper_cost': total_paper_cost,
                            'overhead_cost': overhead_cost,
                            'total_production_cost': total_production_cost
                        }

        if best_overall_result:
            # Prepare orders data for display
            processed_orders = [
                {
                    "Order ID": order.order_id,
                    "Width (mm)": order.width,
                    "Height (mm)": order.height,
                    "Quantity": order.quantity,
                    "Value": f"${order.value:.2f}",
                    "A4 Units": round(order.size_a4_units, 2),
                    "UPS per Sheet": best_overall_result['best_combination']['ups_per_order'].get(order.order_id, 0),
                    "Produced Qty": best_overall_result['best_combination']['produced_qty'].get(order.order_id, 0),
                    "Overs": best_overall_result['best_combination']['produced_qty'].get(order.order_id, 0) - order.quantity
                }
                for order in best_overall_result['best_combination']['orders']
            ]

            # Prepare summary data with the desired order
            summary = {
                "Number of Sheets": best_overall_result['best_combination']['sheet_count'],
                "Total Placed Objects": best_overall_result['total_placed_objects'],
                "Total Unique Kinds": best_overall_result['total_unique_kinds'],
                "GP Per Point": f"${best_overall_result['gp_per_point']:.2f}",
                "Gross Profit": f"${best_overall_result['gross_profit']:.2f}",
                "Total Run Value": f"${best_overall_result['total_run_value']:.2f}",
                "Total Overs Cost": f"${best_overall_result['total_overs_cost']:.2f}",
                "Total Courier Cost": f"${best_overall_result['total_courier_cost']:.2f}",
                "Total Plate Cost": f"${best_overall_result['total_plate_cost']:.2f}",
                "Total Paper Cost": f"${best_overall_result['total_paper_cost']:.2f}",
                "Total Overhead Cost": f"${best_overall_result['overhead_cost']:.2f}",
                "Total Production Cost": f"${best_overall_result['total_production_cost']:.2f}"
            }

            # Calculate Empty Seats as 90 - Total Placed Objects
            empty_seats = 90 - best_overall_result['total_placed_objects']
            summary["Empty Seats"] = empty_seats

            # Handle skipped lines
            formatted_skipped = [
                {
                    "Line": line["Line"],
                    "Error": line["Error"],
                    "Content": line["Content"]
                }
                for line in skipped_lines
            ]

            return render_template(
                'index.html',
                input_text=input_text,
                mode=mode,
                mode_description=mode_description,  # This will no longer be displayed
                orders=processed_orders,
                summary=summary,
                skipped_lines=formatted_skipped
            )
        else:
            return render_template('index.html', error="No valid combinations found.", input_text=input_text)

    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)  # Logs the full error to the console for debugging
        return render_template('index.html', error=f"An error occurred: {e}", input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)