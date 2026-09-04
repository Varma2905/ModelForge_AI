import os
import sys
import asyncio
import shutil
import pandas as pd
import numpy as np

# Adjust sys.path to import from backend
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Now we can import the modules
from app.database.database import db_client
from app.ml.preprocessing import preprocess_dataframe
from app.ml.regression_models import get_regression_model
from app.ml.evaluation import calculate_evaluation_metrics, calculate_statistical_properties
from app.ml.prediction import save_model_package, predict_with_model
from app.visualization.graph_generator import generate_dataset_graphs, generate_regression_graphs
from app.api.ai_routes import run_ai_explanation_pipeline
from app.reports.pdf_generator import build_pdf_report

async def run_pipeline_test():
    print("==================================================")
    print("STARTING E2E BACKEND PIPELINE VERIFICATION")
    print("==================================================")
    
    # Initialize DB
    await db_client.connect()
    print("Database initialized.")
    
    # 1. Create a dummy dataset with missing values and outliers
    raw_data = {
        "Area": [1200, 1500, 1800, 2200, 3000, 1000, 5000, 1600, np.nan, 2100],  # 5000 is an outlier, Nan is missing
        "Bedroom": [2, 3, 3, 4, 5, 2, 10, 3, 3, 4],  # 10 is an outlier
        "Age": [15, 10, 8, 5, 2, 20, 1, 12, np.nan, 6],
        "Price": [3500000, 5000000, 6000000, 7500000, 11000000, 2800000, 20000000, 5200000, 4800000, 7000000]
    }
    
    df = pd.DataFrame(raw_data)
    print(f"Original dataset created: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # 2. Run Preprocessing
    print("\n[STEP 1] Data Preprocessing...")
    preprocess_config = {
        "missing": "mean",
        "dedupe": True,
        "outlier": "iqr",
        "scaling": "standard"
    }
    
    processed_df, prep_summary = preprocess_dataframe(df, preprocess_config)
    print("Preprocessing Completed. Summary:")
    print(prep_summary)
    assert prep_summary["final_rows"] > 0, "Preprocessed dataframe has 0 rows!"
    
    # 3. Simulate training with multiple algorithms
    print("\n[STEP 2] Testing the 9 Regression Engines...")
    algorithms = [
        "LinearRegression", 
        "MultipleLinear", 
        "PolynomialRegression", 
        "Ridge", 
        "Lasso", 
        "ElasticNet", 
        "DecisionTree", 
        "RandomForest", 
        "SVR"
    ]
    
    features = ["Area", "Bedroom", "Age"]
    target = "Price"

    # This script calls internal ML/DB functions directly (bypassing the HTTP
    # auth layer), so it fabricates its own user_id to satisfy the multi-tenancy
    # fields the routes normally populate.
    TEST_USER_ID = "test_user_pipeline"
    TEST_DATASET_ID = "test_dataset_123"
    TEST_MODEL_ID = "test_rf_model"

    # Save the processed dataset in DB to link with models
    dataset_doc = {
        "_id": TEST_DATASET_ID,
        "user_id": TEST_USER_ID,
        "name": "house_prices_test.csv",
        "columns": processed_df.columns.tolist(),
        "rows": processed_df.where(pd.notnull(processed_df), None).values.tolist(),
        "data_types": {col: str(dtype) for col, dtype in zip(processed_df.columns, processed_df.dtypes)},
        "row_count": len(processed_df),
        "col_count": len(processed_df.columns)
    }
    await db_client.insert_one("datasets", dataset_doc)
    
    for algo in algorithms:
        print(f"  Training {algo}...")
        
        # Fit model
        model = get_regression_model(algo, {"random_state": 42} if algo in ["DecisionTree", "RandomForest"] else {})
        # Fit on whole data for test simplicity
        X = processed_df[features]
        y = processed_df[target]
        model.fit(X, y)
        
        # Evaluate model
        y_pred = model.predict(X)
        metrics = calculate_evaluation_metrics(y.values, y_pred, len(X), len(features))
        print(f"    Metrics for {algo}: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}")
        
    # We will pick RandomForest for the rest of the pipeline
    print("\n[STEP 3] Running Full Pipeline for RandomForest...")
    rf_model = get_regression_model("RandomForest", {"n_estimators": 50, "random_state": 42})
    X = processed_df[features]
    y = processed_df[target]
    rf_model.fit(X, y)
    
    # Run OLS analysis
    stats_properties = calculate_statistical_properties(X, y)
    print("  Statistical coefficients calculated.")
    
    # Save model package
    model_id = TEST_MODEL_ID
    save_model_package(model_id, rf_model, features, target, scaler=None)
    print(f"  Model saved to saved_models/{model_id}.pkl.")
    
    # Generate graphs
    print("\n[STEP 4] Generating Graphs...")
    ds_graphs = generate_dataset_graphs(df[features], target, model_id)
    y_pred = rf_model.predict(X)
    reg_graphs = generate_regression_graphs(y.values, y_pred, features, stats_properties["coefficients"], model_id)
    graph_paths = {**ds_graphs, **reg_graphs}
    print("  Generated graphs:")
    for name, path in graph_paths.items():
        print(f"    - {name}: {path}")
        assert os.path.exists(path), f"Graph file {path} was not created!"

    # Save model doc in database
    model_doc = {
        "_id": model_id,
        "user_id": TEST_USER_ID,
        "dataset_id": TEST_DATASET_ID,
        "dataset_name": "house_prices_test.csv",
        "model": "RandomForest",
        "features": features,
        "target": target,
        "split": {"test_size": 0.2, "val_size": 0.0, "random_state": 42},
        "metrics": calculate_evaluation_metrics(y.values, y_pred, len(X), len(features)),
        "statistical_analysis": stats_properties,
        "graph_paths": graph_paths,
        "status": "completed",
        "created_at": pd.Timestamp.now().isoformat()
    }
    await db_client.insert_one("models", model_doc)
    
    # 4. Make prediction
    print("\n[STEP 5] Prediction Engine...")
    test_inputs = {"Area": 0.5, "Bedroom": 0.2, "Age": -0.8} # Scaled values
    prediction = predict_with_model(model_id, test_inputs)
    print(f"  Predicted Value: {prediction:,.2f}")
    
    # 5. Run AI agent explanations
    print("\n[STEP 6] Running AI explanation agent network...")
    ai_explanation = await run_ai_explanation_pipeline(model_id, TEST_USER_ID)
    print("  AI explanation compiled successfully. Executive summary:")
    print(f"  '{ai_explanation['summary']}'")
    assert "full_report" in ai_explanation, "Full report not found in AI response"
    
    # 6. Generate PDF report
    print("\n[STEP 7] PDF Report Generation...")
    # Fetch updated model doc
    updated_model_doc = await db_client.find_one("models", {"_id": model_id})
    output_pdf = os.path.join(backend_dir, "static", "reports", f"report_{model_id}.pdf")
    
    build_pdf_report(
        model_id=model_id,
        model_info=updated_model_doc,
        graph_paths=graph_paths,
        ai_report_markdown=updated_model_doc["ai_explanation"],
        output_pdf_path=output_pdf
    )
    print(f"  PDF Report written to: {output_pdf}")
    assert os.path.exists(output_pdf), f"PDF Report file {output_pdf} was not created!"
    
    print("\n==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! BACKEND PIPELINE OK")
    print("==================================================")
    
    # Clean up test output data files/directories AND the DB records / saved
    # model this script creates — previously only the generated PNGs/PDF were
    # removed, leaving stale "test_rf_model" records behind in db.json indefinitely.
    try:
        # Delete generated test static assets
        static_test_dir = os.path.join(backend_dir, "static", "graphs", model_id)
        if os.path.exists(static_test_dir):
            shutil.rmtree(static_test_dir)
        if os.path.exists(output_pdf):
            os.remove(output_pdf)

        # Delete the saved model package
        saved_model_path = os.path.join(backend_dir, "app", "models", "saved_models", f"{model_id}.pkl")
        if os.path.exists(saved_model_path):
            os.remove(saved_model_path)

        # Delete the DB records this script created
        await db_client.delete_one("datasets", {"_id": TEST_DATASET_ID})
        await db_client.delete_one("models", {"_id": model_id})
        await db_client.delete_one("reports", {"_id": f"rep_{model_id}"})

        print("Test artifacts cleaned up (files + DB records + saved model).")
    except Exception as e:
        print(f"Cleanup warning: {e}")

if __name__ == "__main__":
    asyncio.run(run_pipeline_test())
