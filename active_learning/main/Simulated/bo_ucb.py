import argparse
import joblib
import numpy as np
# Unused imports (matplotlib, pandas, PCA) have been removed.
from sklearn.ensemble import GradientBoostingRegressor
from bayes_opt import BayesianOptimization, UtilityFunction

# --- Global variable for the loaded model ---
gbr_model = None 
N_FEATURES = 8   # Default number of features, can be overridden by args

def black_box_function(**kwargs):
    """
    The function to be optimized by Bayesian Optimization.
    It takes keyword arguments (x0, x1, ..., xN-1), normalizes them so they sum to 1,
    and then predicts using the globally loaded gbr_model.
    """
    global gbr_model, N_FEATURES # Access the global model and N_FEATURES

    if gbr_model is None:
        raise ValueError("Gradient Boosting Regressor model (gbr_model) is not loaded.")

    # Construct the feature vector from kwargs
    # Using a list comprehension for slightly more conciseness
    element_values = [kwargs[f'x{i}'] for i in range(N_FEATURES)]
    element = np.array(element_values) + 1e-8 # Add a small epsilon

    # Normalize the elements so their sum is 1 (representing compositions)
    element_sum = element.sum()
    if element_sum == 0:
        # This case is unlikely with pbounds (0,1) and epsilon, but handled.
        normalized_element = element
    else:
        normalized_element = element / element_sum
    
    # Reshape for sklearn model prediction (expects 2D array)
    prediction = gbr_model.predict(normalized_element.reshape(1, -1))
    return prediction[0] # Return the single prediction value

def constraint_sum_one(**kwargs): # Defined but not directly used by BayesianOptimization in this script
    """
    Example constraint function. Ensures sum of feature values is between 0 and 1.
    Normalization is handled in black_box_function for this script's BO.
    """
    global N_FEATURES
    element_values = [kwargs[f'x{i}'] for i in range(N_FEATURES)]
    current_sum = sum(element_values)
    return 0 <= current_sum <= 1

def main(args):
    """
    Main function to run the Bayesian Optimization process.
    """
    global gbr_model, N_FEATURES # Allow modification of global variables
    N_FEATURES = args.n_features

    # --- Load the pre-trained Gradient Boosting Regressor model ---
    print(f"Loading pre-trained GBR model from: {args.model_file}")
    try:
        gbr_model = joblib.load(args.model_file)
    except FileNotFoundError:
        print(f"Error: Model file '{args.model_file}' not found.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # --- Bayesian Optimization ---
    all_runs_log = [] # To store logs from all seeds

    for seed in range(1, args.n_seeds + 1):
        print(f"\n--- Starting Optimization for Seed {seed}/{args.n_seeds} ---")
        current_seed_log = []
        
        # Define parameter bounds
        pbounds = {f'x{i}': (0, 1) for i in range(N_FEATURES)}

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            verbose=0, # 0: silent, 1: overview, 2: everything
            random_state=seed,
            allow_duplicate_points=True # Handles potential re-suggestion by BO
        )

        # UCB acquisition function. Consider making kappa an arg if tuning is needed.
        acquisition_function = UtilityFunction(kind="ucb") # Default kappa
        
        for iteration_num in range(args.bo_iterations):
            batch_targets = []
            batch_params = []
            
            for _ in range(args.bo_batch_size):
                next_point_to_sample = optimizer.suggest(acquisition_function)
                
                # Ensure the suggested point is not a duplicate of an already evaluated point
                # This prevents re-evaluating the exact same point if suggest() proposes it.
                attempts = 0
                max_attempts = 10 # Prevent infinite loop if space is exhausted or too small
                # Check against optimizer.res which stores all registered points
                while any(np.allclose(np.array(list(p['params'].values())), np.array(list(next_point_to_sample.values()))) for p in optimizer.res) and attempts < max_attempts:
                    next_point_to_sample = optimizer.suggest(acquisition_function)
                    attempts += 1
                
                if attempts == max_attempts and any(np.allclose(np.array(list(p['params'].values())), np.array(list(next_point_to_sample.values()))) for p in optimizer.res):
                    print("Warning: Could not find a unique point after several attempts. Proceeding with potentially duplicate point or point already evaluated.")

                target_value = black_box_function(**next_point_to_sample)
                
                batch_targets.append(target_value)
                batch_params.append(next_point_to_sample)
                current_seed_log.append(target_value)
            
            # Register all evaluated points in the batch
            for i in range(len(batch_params)):
                optimizer.register(params=batch_params[i], target=batch_targets[i])
            
            if batch_targets: # Ensure batch_targets is not empty before calling max()
                print(f"Seed {seed}, Iteration {iteration_num + 1}/{args.bo_iterations}: Max target in batch = {max(batch_targets):.3f}, Best overall = {optimizer.max['target']:.3f}")
            else:
                print(f"Seed {seed}, Iteration {iteration_num + 1}/{args.bo_iterations}: No points in batch. Best overall = {optimizer.max['target']:.3f if optimizer.max else 'N/A'}")


        all_runs_log.append(current_seed_log)
        print(f"--- Seed {seed} completed. Best target found: {optimizer.max['target']:.3f} ---")
        print(f"Best parameters: {optimizer.max['params']}")

    # Save the collected logs
    try:
        np.save(args.output_file, all_runs_log)
        print(f"\nAll optimization runs completed. Results saved to: {args.output_file}")
    except Exception as e:
        print(f"Error saving results to '{args.output_file}': {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bayesian Optimization for Material Composition using Gradient Boosting Regressor")

    parser.add_argument('--model_file', type=str, default='gbr.pkl',
                        help="Path to the pre-trained GradientBoostingRegressor model file (e.g., 'gbr.pkl').")
    parser.add_argument('--output_file', type=str, default='result_ucb.npy',
                        help="Path to save the logs from Bayesian Optimization (e.g., 'result_ucb.npy').")
    parser.add_argument('--n_seeds', type=int, default=1,
                        help="Number of independent Bayesian Optimization runs (seeds).")
    parser.add_argument('--bo_iterations', type=int, default=20,
                        help="Number of Bayesian Optimization iterations per seed.")
    parser.add_argument('--bo_batch_size', type=int, default=10,
                        help="Number of points to suggest and evaluate in each BO iteration (batch size).")
    parser.add_argument('--n_features', type=int, default=8,
                        help="Number of features (compositional elements) for optimization.")
    # Optional: Add --ucb_kappa if you want to control it from command line
    # parser.add_argument('--ucb_kappa', type=float, default=2.576, 
    # help="Kappa value for UCB acquisition function.")

    args = parser.parse_args()
    main(args)