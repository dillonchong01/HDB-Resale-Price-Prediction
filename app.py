import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import re
from Pipeline.utils import compute_nearest_for_address

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)

model = None
category_mappings = None
hdb_features = None
mrt_df = None
mall_df = None

MATURE_ESTATES = [
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH", "CENTRAL",
    "CLEMENTI", "GEYLANG", "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS",
    "QUEENSTOWN", "SERANGOON", "TAMPINES", "TOA PAYOH"
]

# Load XGBoost Model and Datasets on Startup
def load_data():
    global model, category_mappings, hdb_features, mrt_df, mall_df  

    # Load XGBoost Model
    saved = joblib.load("xgboost_model.pkl")
    model = saved["model"]
    category_mappings = saved["categories"]
    print("✓ XGBoost Model loaded")
    
    # Load Datasets
    hdb_features = pd.read_csv('Datasets/HDB_Features.csv')    
    mrt_df = pd.read_csv('Datasets/MRT_LatLong.csv')
    mall_df = pd.read_csv('Datasets/Mall_LatLong.csv')

# Function to Parse Address
def clean_address(address):
    # Convert to Uppercase
    cleaned = address.upper()

    # Remove BLK and BLOCK, Replace AVENUE with AVE, Replace STREET with ST
    cleaned = re.sub(r'\bBLK\b', '', cleaned)
    cleaned = re.sub(r'\bBLOCK\b', '', cleaned)
    cleaned = re.sub(r'\bAVENUE\b', 'AVE', cleaned)
    cleaned = re.sub(r'\bSTREET\b', 'ST', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

# Lookup Address in HDB_Features
def lookup_address_features(cleaned_address, flat_type=None):
    match = hdb_features[hdb_features['Address'] == cleaned_address]
    
    if not match.empty:
        row = match.iloc[0]
        
        # Parse Floor Area Map
        floor_area_map = {}
        if 'Floor Area Map' in row and pd.notna(row['Floor Area Map']):
            try:
                import ast
                floor_area_map = ast.literal_eval(row['Floor Area Map'])
            except:
                floor_area_map = {}
        
        result = {
            'found': True,
            'town': row['Town'],
            'lease_commence_date': int(row['Lease Commence Date']),
            'mature': int(row['Mature']),
            'distance_to_mrt': float(row['Distance_to_MRT']),
            'distance_to_mall': float(row['Distance_to_Mall']),
            'floor_area_map': floor_area_map
        }
        
        # If flat_type is provided, extract specific floor area
        if flat_type and flat_type in floor_area_map:
            result['floor_area'] = floor_area_map[flat_type]
        
        return result
    
    return {'found': False}


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/about')
def about():
    """Serve the about page"""
    return render_template('about.html')


@app.route('/api/lookup-address', methods=['POST'])
def lookup_address():
    """
    Lookup address with flat type to get floor area
    BOTH address AND flat_type are REQUIRED
    """
    try:
        data = request.json
        address = data.get('address', '')
        flat_type = data.get('flat_type', '').strip().upper()
        
        # BOTH are required
        if not address:
            return jsonify({'error': 'Address is required'}), 400
        
        if not flat_type:
            return jsonify({'error': 'Flat type is required'}), 400
        
        # Clean the address
        cleaned = clean_address(address)
        
        # Lookup in database with flat_type
        features = lookup_address_features(cleaned, flat_type)
        
        if features['found']:
            # Calculate remaining lease (99-year lease minus years elapsed)
            remaining_lease = 99 - (2026 - features['lease_commence_date'])
            
            response = {
                'found': True,
                'cleaned_address': cleaned,
                'town': features['town'],
                'remaining_lease': remaining_lease,
                'mature': features['mature'],
                'distance_to_mrt': features['distance_to_mrt'],
                'distance_to_mall': features['distance_to_mall'],
                'floor_area_map': features['floor_area_map']
            }
            
            # Add floor_area if it was extracted for this flat_type
            if 'floor_area' in features:
                response['floor_area'] = features['floor_area']
            else:
                # Flat type not available at this address
                response['floor_area'] = None
            
            return jsonify(response)
        else:
            return jsonify({
                'found': False,
                'cleaned_address': cleaned,
                'message': 'Address not found in database. Distances will be computed.'
            })
    
    except Exception as e:
        print(f"Error in lookup_address: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Expects JSON with all required features
    """
    try:
        data = request.json
        
        # Extract and validate inputs
        address = data.get('address', '').strip()
        town = data.get('town', '').strip().upper()
        flat_type = data.get('flat_type', '').strip().upper()
        floor_area = data.get('floor_area')
        floor_level = data.get('floor_level')
        remaining_lease = data.get('remaining_lease')
        
        # Validation
        if not town or not flat_type:
            return jsonify({'error': 'Town and Flat Type are required'}), 400
        
        if floor_area is None or floor_level is None or remaining_lease is None:
            return jsonify({'error': 'Floor Area, Floor Level, and Remaining Lease are required'}), 400
        
        # Convert to appropriate types
        try:
            floor_area = float(floor_area)
            floor_level = int(floor_level)
            remaining_lease = int(remaining_lease)
        except ValueError:
            return jsonify({'error': 'Invalid numeric values provided'}), 400
        
        # Get current year and month
        now = datetime.now()
        year = now.year
        month = now.month
        
        # Determine if we need to lookup address or compute distances
        cleaned_address = clean_address(address) if address else None
        
        # Initialize distance variables
        distance_to_mrt = None
        distance_to_mall = None
        mature = None
        
        # Try to lookup address in database first
        if cleaned_address:
            features = lookup_address_features(cleaned_address)
            
            if features['found']:
                # Use database values
                mature = features['mature']
                distance_to_mrt = features['distance_to_mrt']
                distance_to_mall = features['distance_to_mall']
            else:
                print(f"Address not found in database: {cleaned_address}")
        
        # If address not found or not provided, compute distances and mature status
        if distance_to_mrt is None or distance_to_mall is None:
            # Determine mature status from town
            mature = 1 if town in MATURE_ESTATES else 0
            
            # Compute distances using utils.py
            if cleaned_address:
                print(f"Computing distances for: {cleaned_address}")
                try:
                    # Use the compute_nearest_for_address function from utils.py
                    result = compute_nearest_for_address(cleaned_address, mrt_df, mall_df)
                    
                    if result:
                        distance_to_mrt = result['Distance_to_MRT']
                        distance_to_mall = result['Distance_to_Mall']
                        print(f"✓ Computed distances - MRT: {distance_to_mrt}km, Mall: {distance_to_mall}km")
                    else:
                        return jsonify({
                            'error': 'Could not geocode address. Please check the address format.'
                        }), 400
                        
                except Exception as e:
                    print(f"Error computing distances: {e}")
                    return jsonify({
                        'error': f'Error computing distances: {str(e)}'
                    }), 500
            else:
                # No address provided and not in database - use default values
                return jsonify({
                    'error': 'Address is required to compute distances for new locations'
                }), 400
        
        features_df = pd.DataFrame([{
            'Town': town,
            'Flat Type': flat_type,
            'Floor Area': floor_area,
            'Year': year,
            'Month': month,
            'Floor Level': floor_level,
            'Remaining Lease': remaining_lease,
            'Mature': mature,
            'Distance_to_MRT': distance_to_mrt,
            'Distance_to_Mall': distance_to_mall
        }])
        
        # Convert categorical features based on category mapping
        for col in category_mappings:
            features_df[col] = pd.Categorical(
                features_df[col],
                categories=category_mappings[col]
            )
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Round to nearest 1000
        rounded_prediction = round(prediction / 1000) * 1000
        
        # Return prediction
        response = {
            'predicted_price': int(rounded_prediction),
            'distance_to_mrt': round(distance_to_mrt, 3) if distance_to_mrt is not None else None,
            'distance_to_mall': round(distance_to_mall, 3) if distance_to_mall is not None else None
        }
        
        print(f"✓ Prediction: SGD ${rounded_prediction:,.0f}")
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'data_loaded': all([
            hdb_features is not None,
            mrt_df is not None,
            mall_df is not None
        ])
    })

load_data()

# if __name__ == '__main__':
#     # Run the Flask app
#     port = int(os.getenv('PORT', 5000))
#     app.run(debug=True, host='0.0.0.0', port=port)