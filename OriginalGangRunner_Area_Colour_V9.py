# OriginalGangRunner_Area_Colour_V9.py
# Optimised for BC .

import pulp
import colorama
from colorama import Fore, Style
import re
import json
import sys
import argparse

# Initialize colorama 
colorama.init(autoreset=True)

# Configuration
def get_configuration(high_priority_mode=False, multiplier=1, min_sheets=245, max_sheets=500, increment=5):
    # Base Objective Function Weight
    base_order_weight = 1000000.0   # Reward for including orders
    
    # Adjust order_weight based on the mode and multiplier
    if high_priority_mode or multiplier != 1:
        order_weight = base_order_weight * multiplier
        print(Fore.YELLOW + f"Order Weight adjusted to {order_weight} (multiplier applied: {multiplier}).")
    else:
        order_weight = base_order_weight
        
    # Maximum Acceptable Overs Percentage
    max_overs_percentage = 500.0  # 500%
    
    # Validate sheet parameters
    if min_sheets <= 0 or max_sheets <= 0 or increment <= 0:
        raise ValueError("Sheet parameters must be positive numbers.")
    if min_sheets > max_sheets:
        raise ValueError("Minimum sheets cannot be greater than maximum sheets.")
        
    # Possible Sheet Counts to Test
    possible_sheets = list(range(int(min_sheets), int(max_sheets) + 1, int(increment)))  # Sheets from min_sheets to max_sheets in increments
    
    # Possible Waste Weights to Test
    possible_waste_weights = [0, 50]  # Adjust this range as needed
    
    # Cost Parameters
    plate_cost_per_run = 80.0          # Base cost per run for plates
    paper_cost_per_sheet = 0.394        # Cost per sheet of paper
    courier_cost_per_order = 12.0       # Courier cost per order
    
    # Validate configuration values
    if plate_cost_per_run <= 0 or paper_cost_per_sheet <= 0 or courier_cost_per_order <= 0:
        raise ValueError("Cost parameters must be positive numbers.")
    if not possible_sheets or not all(s > 0 for s in possible_sheets):
        raise ValueError("Possible sheets must be a list of positive numbers.")
    if not possible_waste_weights or not all(w >= 0 for w in possible_waste_weights):
        raise ValueError("Possible waste weights must be a list of non-negative numbers.")
        
    # Return the configuration as a dictionary
    return {
        'order_weight': order_weight,
        'max_overs_percentage': max_overs_percentage,
        'possible_sheets': possible_sheets,
        'possible_waste_weights': possible_waste_weights,
        'plate_cost_per_run': plate_cost_per_run,
        'paper_cost_per_sheet': paper_cost_per_sheet,
        'courier_cost_per_order': courier_cost_per_order,
    }
    
# Order class
class Order:
    def __init__(self, order_id, width, height, quantity, value, paper_type_description="", base_order_id=None):
        # Input validation
        if width <= 0 or height <= 0:
            raise ValueError(f"Order {order_id}: Width and height must be positive numbers.")
        if quantity <= 0:
            raise ValueError(f"Order {order_id}: Quantity must be a positive number.")
        if value < 0:
            raise ValueError(f"Order {order_id}: Value must be a non-negative number.")

        self.order_id = order_id
        self.base_order_id = base_order_id if base_order_id else order_id  # Track base order ID
        self.width = width          # in mm
        self.height = height        # in mm
        self.quantity = quantity
        self.size_a4_units = 0      # Will be calculated based on size
        self.value = value          # Total value of the order
        self.paper_type_description = paper_type_description
        self.max_ups = 0            # Will be set during precomputation

    def to_dict(self):
        """
        Converts the Order object into a dictionary for easy JSON serialization and template rendering.
        """
        return {
            "Order ID": self.order_id,
            "Width (mm)": self.width,
            "Height (mm)": self.height,
            "Quantity": self.quantity,
            "Value": self.value,
            "A4 Units": round(self.size_a4_units, 2),
            "Paper Type Description": self.paper_type_description,
            "Base Order ID": self.base_order_id,
            "Max UPS": self.max_ups
        }

# Sheet class
class Sheet:
    def __init__(self, width, height):
        # Input validation
        if width <= 0 or height <= 0:
            raise ValueError("Sheet dimensions must be positive numbers.")

        self.width = width          # in mm
        self.height = height        # in mm
        self.capacity_a4_units = 0  # Will be calculated based on size

# Data Importer for Interactive Mode
def parse_input_data(input_data):
    """
    Parses the input data into Order objects.

    Args:
        input_data (str): Multiline string with tab-separated or space-separated order details.

    Returns:
        tuple: (list of Order objects, list of skipped lines with errors)
    """
    orders = []
    skipped_lines = []
    line_number = 0

    # Split the input data into lines
    lines = input_data.strip().split('\n')

    for line in lines:
        line_number += 1
        # Split each line by tabs or multiple spaces
        fields = re.split(r'\t+|\s{2,}', line.strip())

        # Skip empty lines
        if not fields or all(field == '' for field in fields):
            continue

        # Extract relevant fields based on corrected mapping
        try:
            order_id = fields[0].strip()
            date = fields[1].strip()
            quantity_str = fields[2].strip()

            # Description is the 4th field
            description = fields[3].strip() if len(fields) > 3 else ''

            # Size string candidate is the 5th field
            size_str_candidate = fields[4].strip() if len(fields) > 4 else ''

            orientation = fields[5].strip() if len(fields) > 5 else ''
            value_str = fields[6].strip() if len(fields) > 6 else ''
            comments = fields[7].strip() if len(fields) > 7 else ''  # Adjusted index

            # Initialize size variables
            width = None
            height = None

            # Determine if "Final Trim to" is present in the description
            final_trim_match = re.search(r'Final Trim to\s*(\d+(?:\.\d+)?)mmx(\d+(?:\.\d+)?)mm', description, re.IGNORECASE)
            if final_trim_match:
                width = float(final_trim_match.group(1)) + 4  # add bleed
                height = float(final_trim_match.group(2)) + 4  # add bleed
            else:
                # Attempt to extract size from size_str_candidate
                size_match = re.search(r'(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)', size_str_candidate)
                if size_match:
                    width = float(size_match.group(1)) + 4  # add bleed
                    height = float(size_match.group(2)) + 4  # add bleed
                else:
                    # If size not found, search the entire line
                    size_match = re.search(r'(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)mm', line)
                    if size_match:
                        width = float(size_match.group(1)) + 2  # add bleed
                        height = float(size_match.group(2)) + 2  # add bleed
                    else:
                        # Unable to find size
                        skipped_lines.append({"Line": line_number, "Error": "Invalid size format: Unable to extract size.", "Content": line})
                        continue

            # Extract value
            if value_str.upper() == 'N/A':
                value = 0.0
                # Do not add to warnings or skipped_lines
            else:
                try:
                    value = float(value_str)
                except ValueError:
                    skipped_lines.append({"Line": line_number, "Error": f"Invalid value: '{value_str}'", "Content": line})
                    continue

            # Parse quantity
            try:
                quantity = int(quantity_str)
            except ValueError:
                skipped_lines.append({"Line": line_number, "Error": f"Invalid quantity: '{quantity_str}'", "Content": line})
                continue

            # **Extract number of kinds from description using regex**
            # Use re.match to ensure it starts with "X kinds of"
            kinds_match = re.match(r'^(\d+)\s+kinds?\s+of', description, re.IGNORECASE)
            if kinds_match:
                num_kinds = int(kinds_match.group(1))
            else:
                # Default to 1 if description does not start with "X kinds of"
                num_kinds = 1

            # Generate orders based on number of kinds
            for kind in range(1, num_kinds + 1):
                if num_kinds > 1:
                    new_order_id = f"{order_id}_{kind}"
                else:
                    new_order_id = order_id

                # Calculate value per kind
                value_per_kind = value / num_kinds if num_kinds > 0 else 0.0

                # Create an Order instance with divided value and base_order_id
                order_instance = Order(new_order_id, width, height, quantity, value_per_kind, paper_type_description="", base_order_id=order_id)
                orders.append(order_instance)

        except Exception as e:
            skipped_lines.append({"Line": line_number, "Error": f"Unexpected error: {e}", "Content": line})
            continue

    return orders, skipped_lines

# Data Importer for JSON Mode
def parse_json_input(json_data):
    """
    Parses the JSON input data into Order objects.

    Args:
        json_data (dict or list): JSON data containing order details.

    Returns:
        tuple: (list of Order objects, list of skipped lines with errors)
    """
    orders = []
    skipped_lines = []
    try:
        for order_entry in json_data:
            order_id = order_entry.get("order_id")
            quantity = order_entry.get("quantity")
            kinds = order_entry.get("kinds", 1)
            price = order_entry.get("price", 0.0)
            final_trim_width_mm = order_entry.get("final_trim_width_mm")
            final_trim_height_mm = order_entry.get("final_trim_height_mm")
            paper_type_description = order_entry.get("paper_type_description", "")

            if not all([order_id, quantity, final_trim_width_mm, final_trim_height_mm]):
                skipped_lines.append({"Line": "N/A", "Error": "Missing required fields.", "Content": order_entry})
                continue

            # Generate orders based on number of kinds
            for kind in range(1, kinds + 1):
                if kinds > 1:
                    new_order_id = f"{order_id}_{kind}"
                else:
                    new_order_id = order_id

                # Calculate value per kind
                value_per_kind = price / kinds if kinds > 0 else 0.0

                # Create an Order instance
                order_instance = Order(
                    order_id=new_order_id,
                    width=final_trim_width_mm,
                    height=final_trim_height_mm,
                    quantity=quantity,
                    value=value_per_kind,
                    paper_type_description=paper_type_description,
                    base_order_id=order_id
                )
                orders.append(order_instance)
    except Exception as e:
        skipped_lines.append({"Line": "N/A", "Error": f"Error parsing JSON input: {e}", "Content": json_data})

    return orders, skipped_lines

# Function to calculate A4 units for each order
def calculate_a4_units(order):
    # Calculate the area of the order
    order_area = order.width * order.height
    # Area of an A4 sheet
    a4_area = 210 * 297
    # Calculate size in A4 units
    order.size_a4_units = order_area / a4_area

# Function to calculate sheet capacity in A4 units
def calculate_sheet_capacity(sheet):
    # Calculate the area of the sheet
    sheet_area = sheet.width * sheet.height
    # Area of an A4 sheet
    a4_area = 210 * 297
    # Calculate capacity in A4 units
    sheet.capacity_a4_units = sheet_area / a4_area

# Function to find the best combination of orders using ILP with precomputed max_ups
def find_best_combination_ilp(orders, sheet, sheets_fixed, config):
    try:
        # Unpack configuration parameters
        order_weight = config['order_weight']
        waste_weight = config.get('waste_weight', 0)  # Default to 0 if not set
        max_overs_percentage = config['max_overs_percentage']

        # Precompute max_ups and filter valid orders
        valid_orders = []
        for order in orders:
            if order.size_a4_units <= 0:
                continue  # Exclude invalid orders
            max_ups = int(sheet.capacity_a4_units // order.size_a4_units)
            if max_ups > 0:
                order.max_ups = max_ups
                valid_orders.append(order)
            # Orders with max_ups == 0 are implicitly excluded

        if not valid_orders:
            return None  # No valid orders to process

        # Create the model
        model = pulp.LpProblem("Gang_Run_Optimization", pulp.LpMaximize)

        # Decision variables
        y = {}    # Binary variable: 1 if order is included, 0 otherwise
        ups = {}  # Number of ups per sheet for each order
        produced_qty = {}  # Total produced quantity for each order

        for order in valid_orders:
            y[order.order_id] = pulp.LpVariable(f"y_{order.order_id}", cat='Binary')
            ups[order.order_id] = pulp.LpVariable(f"ups_{order.order_id}", lowBound=0, upBound=order.max_ups, cat='Integer')
            produced_qty[order.order_id] = pulp.LpVariable(f"produced_qty_{order.order_id}", lowBound=order.quantity * y[order.order_id].upBound, cat='Integer')

        # Objective: Maximize total orders included, penalize overs
        total_orders_included = pulp.lpSum([y[order.order_id] for order in valid_orders])
        total_deviation = pulp.lpSum([
            produced_qty[order.order_id] - order.quantity * y[order.order_id]
            for order in valid_orders
        ])

        model += order_weight * total_orders_included - waste_weight * total_deviation

        # Constraints
        for order in valid_orders:
            # Ensure produced quantity meets or exceeds ordered quantity
            model += produced_qty[order.order_id] >= order.quantity * y[order.order_id], f"Min_Production_Order_{order.order_id}"

            # Production must not exceed ordered quantity by more than the acceptable percentage
            model += produced_qty[order.order_id] <= order.quantity * (1 + max_overs_percentage / 100) * y[order.order_id], f"Max_Production_Order_{order.order_id}"

            # Link produced quantity to sheets and ups
            model += produced_qty[order.order_id] == sheets_fixed * ups[order.order_id], f"Link_Produced_Ups_Order_{order.order_id}"

        # Sheet capacity constraint
        model += pulp.lpSum([
            ups[order.order_id] * order.size_a4_units
            for order in valid_orders
        ]) <= sheet.capacity_a4_units, "Sheet_Capacity_Constraint"

        # Solve the model
        solver = pulp.PULP_CBC_CMD(msg=False, threads=4, timeLimit=300)  # 5-minute limit
        model.solve(solver)

        if model.status == pulp.LpStatusOptimal:
            best_combination = {
                'orders': [order for order in valid_orders if y[order.order_id].varValue == 1],
                'sheet_count': sheets_fixed,
                'ups_per_order': {order.order_id: int(ups[order.order_id].varValue) for order in valid_orders if y[order.order_id].varValue == 1},
                'produced_qty': {order.order_id: int(produced_qty[order.order_id].varValue) for order in valid_orders if y[order.order_id].varValue == 1},
                'total_overs': sum(
                    int(produced_qty[order.order_id].varValue) - order.quantity 
                    for order in valid_orders if y[order.order_id].varValue == 1
                ),
                'under_utilized_space_per_sheet': sheet.capacity_a4_units - sum(
                    int(ups[order.order_id].varValue) * order.size_a4_units
                    for order in valid_orders if y[order.order_id].varValue == 1
                ),
                'total_utilized_space_per_sheet': sum(
                    int(ups[order.order_id].varValue) * order.size_a4_units
                    for order in valid_orders if y[order.order_id].varValue == 1
                ),
            }
            return best_combination
        else:
            return None
    except Exception as e:
        print(Fore.RED + f"An error occurred during optimization: {e}")
        return None

# Function to generate the output
def generate_output(best_result, sheet, orders, config, json_mode=False):
    best_combination = best_result['best_combination']
    if not best_combination:
        print(Fore.RED + "No valid combinations found.")
        return

    plate_cost_per_run = config['plate_cost_per_run']
    paper_cost_per_sheet = config['paper_cost_per_sheet']
    courier_cost_per_order = config['courier_cost_per_order']

    if json_mode:
        # Prepare JSON output
        output = {
            "Optimized_Gang_Run": {
                "Waste_Weight_Used": best_result['waste_weight'],
                "Sheet_Size_mm": {
                    "width": sheet.width,
                    "height": sheet.height
                },
                "Number_of_Sheets": best_combination['sheet_count'],
                "Orders_Included": []
            }
        }

        total_run_value = 0
        total_overs_cost = 0
        total_equivalent_a4s = 0  # To sum up the total equivalent A4s used
        total_unique_kinds = best_result.get('total_unique_kinds', len(best_combination['orders']))
        total_placed_objects = best_result.get('total_placed_objects', sum(best_combination['ups_per_order'].values()))

        for order in best_combination['orders']:
            ups = best_combination['ups_per_order'][order.order_id]
            produced_qty = best_combination['produced_qty'][order.order_id]
            overs = produced_qty - order.quantity
            unit_cost = (config['paper_cost_per_sheet'] * order.size_a4_units)  # Define unit_cost based on paper usage
            overs_cost = overs * unit_cost  # Calculate overs_cost based on actual cost

            total_run_value += order.value
            total_overs_cost += overs_cost
            total_equivalent_a4s += round(order.size_a4_units * ups, 2)

            output["Optimized_Gang_Run"]["Orders_Included"].append({
                "order_id": order.order_id,
                "size_mm": {"width": order.width, "height": order.height},
                "equivalent_a4s": round(order.size_a4_units * ups, 2),
                "ups_per_sheet": ups,
                "ordered_qty": order.quantity,
                "produced_qty": produced_qty,
                "overs": overs,
                "order_value": round(order.value, 2),
                "overs_cost": round(overs_cost, 2)
            })

        # Calculate Costs
        total_plate_cost = plate_cost_per_run
        total_paper_cost = best_combination['sheet_count'] * paper_cost_per_sheet

        # Waste paper cost calculation
        total_waste_a4_units = best_combination['under_utilized_space_per_sheet'] * best_combination['sheet_count']
        cost_per_a4_unit = paper_cost_per_sheet / sheet.capacity_a4_units
        total_waste_paper_cost = total_waste_a4_units * cost_per_a4_unit

        # **Courier Cost Calculation**
        # Calculate the number of unique base_order_ids
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

        # Add Summary to JSON
        output["Optimized_Gang_Run"]["Summary"] = {
            "Total_Run_Value": round(total_run_value, 2),
            "Total_Overs_Cost": round(total_overs_cost, 2),
            "Total_Courier_Cost": round(total_courier_cost, 2),
            "Total_Equivalent_A4s_Used": round(total_equivalent_a4s, 2),
            "Total_Unique_Kinds": total_unique_kinds,
            "Total_Placed_Objects": total_placed_objects,
            "Total_Waste_Paper_A4_Units": round(total_waste_a4_units, 2),
            "Total_Waste_Paper_Cost": round(total_waste_paper_cost, 2),
            "Total_Plate_Cost": round(total_plate_cost, 2),
            "Total_Paper_Cost": round(total_paper_cost, 2),
            "Total_Overhead_Cost": round(overhead_cost, 2),
            "Total_Production_Cost": round(total_production_cost, 2),
            "Gross_Profit": round(gross_profit, 2),
            "Points_Per_Run": points_per_run,
            "GP_Per_Point": round(gp_per_point, 2)
        }

        # Print JSON output
        print(json.dumps(output, indent=4))

    else:
        # Interactive Mode Output (Existing Functionality)
        # [Command-line interaction logic]
        # For Flask, this is handled separately
        pass  # Placeholder for command-line functionality

# Function to generate JSON output
def generate_json_output(best_result, sheet, orders, config):
    # This function is now integrated within generate_output for better control
    pass  # Placeholder if needed separately

# Main function to execute the program
def main():
    parser = argparse.ArgumentParser(description="Process orders for optimization.")
    parser.add_argument('--json', action='store_true', help='Run the script in JSON mode.')
    parser.add_argument('--input', type=str, help='Path to the JSON input file.')
    parser.add_argument('--output', type=str, help='Path to save the JSON output file.')
    args = parser.parse_args()

    try:
        if args.json:
            # JSON Mode
            if not args.input:
                print(Fore.RED + "JSON mode requires an input file. Use --input to specify the JSON file.")
                sys.exit(1)

            # Read JSON data from the input file
            with open(args.input, 'r') as f:
                try:
                    json_data = json.load(f)
                except json.JSONDecodeError as e:
                    print(Fore.RED + f"Invalid JSON format: {e}")
                    sys.exit(1)

            # Parse JSON data into Order objects
            orders_list, skipped_lines = parse_json_input(json_data)

            if not orders_list:
                print(Fore.RED + "No valid orders found in JSON input.")
                sys.exit(1)

            # Calculate A4 units for each order
            for order in orders_list:
                calculate_a4_units(order)

            # Define sheet size (e.g., 590mm x 847mm)
            sheet = Sheet(590, 847)
            calculate_sheet_capacity(sheet)

            # Get configuration (assuming Normal Mode for JSON)
            config = get_configuration()

            best_overall_result = None
            max_gp_per_point = float('-inf')  # Initialize max GP per Point

            # Iterate over possible waste weights and sheet counts
            for waste_weight in config['possible_waste_weights']:
                config['waste_weight'] = waste_weight  # Set the current waste weight

                for sheets_fixed in config['possible_sheets']:
                    best_combination = find_best_combination_ilp(orders_list, sheet, sheets_fixed, config)
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
                            }

            if best_overall_result:
                # Generate JSON Output
                best_combination = best_overall_result['best_combination']
                output = {
                    "Optimized_Gang_Run": {
                        "Waste_Weight_Used": best_overall_result['waste_weight'],
                        "Sheet_Size_mm": {
                            "width": sheet.width,
                            "height": sheet.height
                        },
                        "Number_of_Sheets": best_combination['sheet_count'],
                        "Orders_Included": []
                    }
                }

                total_run_value = 0
                total_overs_cost = 0
                total_equivalent_a4s = 0  # To sum up the total equivalent A4s used
                total_unique_kinds = best_overall_result.get('total_unique_kinds', len(best_combination['orders']))
                total_placed_objects = best_overall_result.get('total_placed_objects', sum(best_combination['ups_per_order'].values()))

                for order in best_combination['orders']:
                    ups = best_combination['ups_per_order'][order.order_id]
                    produced_qty = best_combination['produced_qty'][order.order_id]
                    overs = produced_qty - order.quantity
                    unit_cost = (config['paper_cost_per_sheet'] * order.size_a4_units)  # Define unit_cost based on paper usage
                    overs_cost = overs * unit_cost  # Calculate overs_cost based on actual cost

                    total_run_value += order.value
                    total_overs_cost += overs_cost
                    total_equivalent_a4s += round(order.size_a4_units * ups, 2)

                    output["Optimized_Gang_Run"]["Orders_Included"].append({
                        "order_id": order.order_id,
                        "size_mm": {"width": order.width, "height": order.height},
                        "equivalent_a4s": round(order.size_a4_units * ups, 2),
                        "ups_per_sheet": ups,
                        "ordered_qty": order.quantity,
                        "produced_qty": produced_qty,
                        "overs": overs,
                        "order_value": round(order.value, 2),
                        "overs_cost": round(overs_cost, 2)
                    })

                # Calculate Costs
                total_plate_cost = config['plate_cost_per_run']
                total_paper_cost = best_combination['sheet_count'] * config['paper_cost_per_sheet']

                # Waste paper cost calculation
                total_waste_a4_units = best_combination['under_utilized_space_per_sheet'] * best_combination['sheet_count']
                cost_per_a4_unit = config['paper_cost_per_sheet'] / sheet.capacity_a4_units
                total_waste_paper_cost = total_waste_a4_units * cost_per_a4_unit

                # **Courier Cost Calculation**
                unique_base_order_ids = set(order.base_order_id for order in best_combination['orders'])
                total_courier_cost = len(unique_base_order_ids) * config['courier_cost_per_order']

                # **Calculate Overhead Costs**
                overhead_cost = 0.09 * total_run_value  # 9% of total run value

                # **Update Total Production Cost to include Overhead Costs**
                total_production_cost = total_plate_cost + total_paper_cost + total_courier_cost + overhead_cost  # Excluding overs_cost

                # **Calculate Gross Profit by excluding overs_cost**
                gross_profit = total_run_value - total_production_cost - total_overs_cost  # Keep as is after correcting overs_cost

                # Calculate Points per Run and GP per Point
                points_per_run = 1500 + best_combination['sheet_count']
                gp_per_point = gross_profit / points_per_run if points_per_run != 0 else 0

                # Add Summary to JSON
                output["Optimized_Gang_Run"]["Summary"] = {
                    "Total_Run_Value": round(total_run_value, 2),
                    "Total_Overs_Cost": round(total_overs_cost, 2),
                    "Total_Courier_Cost": round(total_courier_cost, 2),
                    "Total_Equivalent_A4s_Used": round(total_equivalent_a4s, 2),
                    "Total_Unique_Kinds": total_unique_kinds,
                    "Total_Placed_Objects": total_placed_objects,
                    "Total_Waste_Paper_A4_Units": round(total_waste_a4_units, 2),
                    "Total_Waste_Paper_Cost": round(total_waste_paper_cost, 2),
                    "Total_Plate_Cost": round(total_plate_cost, 2),
                    "Total_Paper_Cost": round(total_paper_cost, 2),
                    "Total_Overhead_Cost": round(overhead_cost, 2),
                    "Total_Production_Cost": round(total_production_cost, 2),
                    "Gross_Profit": round(gross_profit, 2),
                    "Points_Per_Run": points_per_run,
                    "GP_Per_Point": round(gp_per_point, 2)
                }

                # Output JSON
                if args.output:
                    with open(args.output, 'w') as outfile:
                        json.dump(output, outfile, indent=4)
                    print(Fore.GREEN + f"JSON output saved to {args.output}")
                else:
                    print(json.dumps(output, indent=4))
        else:
            # Interactive Mode Output (Existing Functionality)
            # [Command-line interaction logic]
            # For Flask, this is handled separately
            pass  # Placeholder for command-line functionality

    except Exception as e:
        print(Fore.RED + f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# OLD DATA IMPORTER (for reference)
# Data Importer
def parse_input_data_old(input_data):
    """
    Parses the input data into Order objects.

    Args:
        input_data (str): Multiline string with tab-separated order details.

    Returns:
        tuple: (list of Order objects, list of formatted order strings, list of skipped lines with errors)
    """
    orders = []
    formatted_orders_output = []
    skipped_lines = []
    line_number = 0

    # Split the input data into lines
    lines = input_data.strip().split('\n')

    for line in lines:
        line_number += 1
        # Split each line by tabs or multiple spaces
        fields = re.split(r'\t+|\s{2,}', line.strip())

        # Check if there are at least 7 fields
        if len(fields) < 7:
            skipped_lines.append((line_number, line, "Insufficient number of fields"))
            continue

        # Extract relevant fields (first 7)
        order_id = fields[0].strip()
        date = fields[1].strip()  # Date is not used in output
        quantity_str = fields[2].strip()
        description = fields[3].strip()
        size_str = fields[4].strip()
        orientation = fields[5].strip()
        value_str = fields[6].strip()

        # Parse quantity
        try:
            quantity = int(quantity_str)
        except ValueError:
            skipped_lines.append((line_number, line, f"Invalid quantity: '{quantity_str}'"))
            continue

        # Parse value
        try:
            value = float(value_str)
        except ValueError:
            skipped_lines.append((line_number, line, f"Invalid value: '{value_str}'"))
            continue

        # Parse number of kinds from description
        kinds_match = re.search(r'(\d+)\s+kinds?\s+of', description, re.IGNORECASE)
        if kinds_match:
            num_kinds = int(kinds_match.group(1))
        else:
            num_kinds = 1  # Default to 1 if not specified

        # Parse size to extract width and height
        size_match = re.search(r'(\d+)\s*[xX]\s*(\d+)', size_str)
        if size_match:
            width = int(size_match.group(1)) + 2  # add bleed
            height = int(size_match.group(2)) + 2  # add bleed
        else:
            skipped_lines.append((line_number, line, f"Invalid size format: '{size_str}'"))
            continue

        # Generate orders based on number of kinds
        for kind in range(1, num_kinds + 1):
            if num_kinds > 1:
                new_order_id = f"{order_id}_{kind}"
            else:
                new_order_id = order_id

            # Calculate value per kind
            value_per_kind = value / num_kinds

            # Create an Order instance with divided value and base_order_id
            order_instance = Order(new_order_id, width, height, quantity, value_per_kind, base_order_id=order_id)
            orders.append(order_instance)

            # Create formatted string for output with divided value
            order_entry = f'    Order("{new_order_id}", {width}, {height}, {quantity}, value={value_per_kind:.2f}),'
            formatted_orders_output.append(order_entry)

    return orders, formatted_orders_output, skipped_lines