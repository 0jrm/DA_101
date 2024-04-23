import numpy as np
from scipy.optimize import minimize

# BLUE
def blue_update(state, covariance, observation, observation_covariance, observation_matrix):
    """
    Perform a BLUE (Best Linear Unbiased Estimator) update on the state estimate.
    
    Parameters:
        state (np.array): The prior state estimate, vector of dimension (n,).
        covariance (np.array): The covariance matrix of the prior state estimate, square matrix of dimension (n, n).
        observation (np.array): The new observation vector, vector of dimension (m,).
        observation_covariance (np.array): The covariance matrix of the observation error, square matrix of dimension (m, m).
        observation_matrix (np.array): The matrix that maps the state space to the observation space, dimension (m, n).
    
    Returns:
        np.array: The updated state estimate.
        np.array: The updated covariance matrix.
    
    Assumes:
        - Errors are Gaussian and errors in estimates and observations are uncorrelated.
        - Matrices and vectors are well-defined and dimensionally consistent.
    """
    # Dimension checks
    n = state.shape[0]
    m = observation.shape[0]
    assert covariance.shape == (n, n), "Covariance matrix dimension mismatch"
    assert observation_covariance.shape == (m, m), "Observation covariance matrix dimension mismatch"
    assert observation_matrix.shape == (m, n), "Observation matrix dimension mismatch"

    # Compute the Kalman gain
    H = observation_matrix
    HT = H.T
    S = H @ covariance @ HT + observation_covariance
    
    try:
        # Numerically stable inversion using pseudo-inverse or Cholesky decomposition
        S_inv = np.linalg.inv(S) if np.all(np.linalg.eigvals(S) > 0) else np.linalg.pinv(S)
        K = covariance @ HT @ S_inv
    except np.linalg.LinAlgError:
        raise ValueError("Singular matrix in inversion, check your input data or model assumptions.")
    
    # Update the state estimate
    y = observation
    x = state
    updated_state = x + K @ (y - H @ x)
    
    # Update the covariance matrix
    updated_covariance = (np.eye(n) - K @ H) @ covariance
    
    return updated_state, updated_covariance

# Kalman Filter
def kf_predict(state, covariance, transition_matrix, process_noise):
    """
    Predict step of the Kalman Filter.

    Parameters:
        state (np.array): The prior state estimate.
        covariance (np.array): The prior covariance matrix.
        transition_matrix (np.array): The state transition matrix.
        process_noise (np.array): The process noise covariance matrix.
    
    Returns:
        np.array: The predicted state.
        np.array: The predicted covariance matrix.
    """
    # Predict the next state
    predicted_state = transition_matrix @ state
    # Predict the next covariance
    predicted_covariance = transition_matrix @ covariance @ transition_matrix.T + process_noise
    
    return predicted_state, predicted_covariance

def kf_update(predicted_state, predicted_covariance, observation, observation_matrix, observation_noise):
    """
    Update step of the Kalman Filter.

    Parameters:
        predicted_state (np.array): The state estimate after prediction.
        predicted_covariance (np.array): The covariance matrix after prediction.
        observation (np.array): The new observation vector.
        observation_matrix (np.array): The matrix that maps the state space to the observation space.
        observation_noise (np.array): The observation noise covariance matrix.
    
    Returns:
        np.array: The updated state estimate.
        np.array: The updated covariance matrix.
    """
    # Compute the Kalman Gain
    H = observation_matrix
    S = H @ predicted_covariance @ H.T + observation_noise  # Innovation (or residual) covariance
    # Compute K using np.linalg.solve to solve SK^T = predicted_covariance @ H^T
    # Transpose the right hand side to match the dimensions for solving:
    rhs = predicted_covariance @ H.T  # This is (n x m)

    # Now solve for K^T using np.linalg.solve:
    K_transpose = np.linalg.solve(S, rhs)  # Solving for (m x m) K^T = (n x m)

    # Transpose back to get K:
    K = K_transpose.T  # Now K is (n x m)
    
    # Update the state
    updated_state = predicted_state + K @ (observation - H @ predicted_state)
    
    # Update the covariance
    n = predicted_covariance.shape[0]
    updated_covariance = (np.eye(n) - K @ H) @ predicted_covariance
    
    return updated_state, updated_covariance

# Extended Kalman Filter
def ekf_predict(state, covariance, transition_function, jacobian_F, process_noise):
    """
    Predict step of the Extended Kalman Filter.

    Parameters:
        state (np.array): The prior state estimate.
        covariance (np.array): The prior covariance matrix.
        transition_function (callable): Nonlinear state transition function.
        jacobian_F (np.array): Jacobian of the transition function evaluated at the current state.
        process_noise (np.array): The process noise covariance matrix.
    
    Returns:
        np.array: The predicted state.
        np.array: The predicted covariance matrix.
    """
    # Predict the next state using the nonlinear transition function
    predicted_state = transition_function(state)
    # Predict the next covariance
    predicted_covariance = jacobian_F @ covariance @ jacobian_F.T + process_noise
    
    return predicted_state, predicted_covariance

def ekf_update(predicted_state, predicted_covariance, observation, observation_function, jacobian_H, observation_noise):
    """
    Update step of the Extended Kalman Filter.

    Parameters:
        predicted_state (np.array): The state estimate after prediction.
        predicted_covariance (np.array): The covariance matrix after prediction.
        observation (np.array): The new observation vector.
        observation_function (callable): Nonlinear observation function.
        jacobian_H (np.array): Jacobian of the observation function evaluated at the predicted state.
        observation_noise (np.array): The observation noise covariance matrix.
    
    Returns:
        np.array: The updated state estimate.
        np.array: The updated covariance matrix.
    """
    # Compute the Kalman Gain
    S = jacobian_H @ predicted_covariance @ jacobian_H.T + observation_noise  # Innovation (or residual) covariance
    K = predicted_covariance @ jacobian_H.T @ np.linalg.inv(S)
    
    # Update the state using the nonlinear observation function
    innovation = observation - observation_function(predicted_state)
    updated_state = predicted_state + K @ innovation
    
    # Update the covariance
    n = predicted_covariance.shape[0]
    updated_covariance = (np.eye(n) - K @ jacobian_H) @ predicted_covariance
    
    return updated_state, updated_covariance

# Ensemble Kalman Filters
def initialize_ensemble(mean_state, covariance, ensemble_size):
    """
    Initializes an ensemble of state estimates from a Gaussian distribution defined by the mean and covariance.

    Parameters:
        mean_state (np.array): The mean state vector from which to generate the ensemble.
        covariance (np.array): The covariance matrix defining the spread of the ensemble.
        ensemble_size (int): The number of ensemble members.

    Returns:
        np.array: An array of shape (ensemble_size, len(mean_state)) representing the ensemble of initial state estimates.
    """
    return np.random.multivariate_normal(mean_state, covariance, ensemble_size)

def forecast_ensemble(ensemble, model_function):
    """
    Propagates each ensemble member forward using the provided model function.

    Parameters:
        ensemble (np.array): Current ensemble of state estimates.
        model_function (callable): The model function to propagate the state.

    Returns:
        np.array: The forecasted ensemble.
    """
    return np.array([model_function(member) for member in ensemble])

def update_ensemble(ensemble, observations, observation_matrix, observation_noise):
    """
    Updates each ensemble member based on the observations using the Kalman Filter update equations.

    Parameters:
        ensemble (np.array): The forecasted ensemble of state estimates.
        observations (np.array): The observed state vector.
        observation_matrix (np.array): The matrix that maps the state space to the observation space.
        observation_noise (np.array): The observation noise covariance matrix.

    Returns:
        np.array: The updated ensemble.
    """
    # Calculate ensemble mean and covariance
    ensemble_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ensemble_mean
    ensemble_covariance = anomalies.T @ anomalies / (len(ensemble) - 1)
    
    # Calculate Kalman Gain
    H = observation_matrix
    R = observation_noise
    S = H @ ensemble_covariance @ H.T + R
    K = ensemble_covariance @ H.T @ np.linalg.inv(S)
    
    # Update each ensemble member
    updated_ensemble = np.zeros_like(ensemble)
    for i, member in enumerate(ensemble):
        innovation = observations - H @ member
        updated_ensemble[i] = member + K @ innovation

    return updated_ensemble

# Localization and inflation
def apply_localization(covariance, localization_function):
    """ Apply a localization function to the covariance matrix. """
    for i in range(len(covariance)):
        for j in range(len(covariance[i])):
            covariance[i, j] *= localization_function(i, j)
    return covariance

def apply_inflation(covariance, inflation_factor):
    """ Inflate the covariance matrix by a constant factor. """
    return covariance * inflation_factor

def stochastic_ensemble_update(ensemble, observations, observation_matrix, observation_noise, process_noise):
    """ Stochastic update of the ensemble. """
    ensemble_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ensemble_mean
    ensemble_covariance = anomalies.T @ anomalies / (len(ensemble) - 1)
    ensemble_covariance = apply_inflation(ensemble_covariance, 1.1)  # Example inflation
    ensemble_covariance = apply_localization(ensemble_covariance, lambda i, j: np.exp(-abs(i-j)/10))  # Example localization
    
    H = observation_matrix
    R = observation_noise
    S = H @ ensemble_covariance @ H.T + R
    K = ensemble_covariance @ H.T @ np.linalg.inv(S)
    
    updated_ensemble = np.zeros_like(ensemble)
    for i, member in enumerate(ensemble):
        perturbed_observation = observations + np.random.multivariate_normal(np.zeros(len(observations)), process_noise)
        innovation = perturbed_observation - H @ member
        updated_ensemble[i] = member + K @ innovation

    return updated_ensemble

def deterministic_ensemble_update(ensemble, observations, observation_matrix, observation_noise):
    """ Deterministic update of the ensemble. """
    ensemble_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ensemble_mean
    ensemble_covariance = anomalies.T @ anomalies / (len(ensemble) - 1)
    ensemble_covariance = apply_inflation(ensemble_covariance, 1.05)  # Example inflation
    ensemble_covariance = apply_localization(ensemble_covariance, lambda i, j: np.exp(-abs(i-j)/5))  # Example localization
    
    H = observation_matrix
    R = observation_noise
    S = H @ ensemble_covariance @ H.T + R
    K = ensemble_covariance @ H.T @ np.linalg.inv(S)
    
    updated_ensemble = np.zeros_like(ensemble)
    for i, member in enumerate(ensemble):
        innovation = observations - H @ member
        updated_ensemble[i] = member + K @ innovation

    return updated_ensemble

def ensemble_kalman_filter(ensemble, observations, model_function, observation_matrix, observation_noise,
                           process_noise=None, deterministic=False, inflation_factor=1.0, localization_function=None):
    """
    Generalized Ensemble Kalman Filter with options for stochastic/deterministic updates, inflation, and localization.

    Parameters:
        ensemble (np.array): Current ensemble of state estimates.
        observations (np.array): Observed state vector.
        model_function (callable): Function to propagate the state.
        observation_matrix (np.array): Matrix that maps the state space to the observation space.
        observation_noise (np.array): Observation noise covariance matrix.
        process_noise (np.array, optional): Process noise covariance matrix; required for stochastic updates.
        deterministic (bool): Whether to use a deterministic update method.
        inflation_factor (float): Factor to inflate the forecast covariance.
        localization_function (callable, optional): Function to apply covariance localization.

    Returns:
        np.array: Updated ensemble.
    """
    
    # Helper functions for localization (example)
    def apply_localization(covariance, localization_func):
        """ Applies a localization function to the covariance matrix element-wise. """
        size = len(covariance)
        for i in range(size):
            for j in range(size):
                covariance[i, j] *= localization_func(i, j)
        return covariance

    # Forecast step
    if process_noise is not None:
        # Stochastic forecast including process noise
        ensemble = np.array([model_function(member) + np.random.multivariate_normal(np.zeros(len(member)), process_noise) for member in ensemble])
    else:
        # Deterministic forecast without explicit process noise
        ensemble = np.array([model_function(member) for member in ensemble])

    # Update step
    ensemble_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ensemble_mean
    ensemble_covariance = anomalies.T @ anomalies / (len(ensemble) - 1)

    # Apply inflation
    ensemble_covariance *= inflation_factor

    # Apply localization if a function is provided
    if localization_function:
        ensemble_covariance = apply_localization(ensemble_covariance, localization_function)

    H = observation_matrix
    R = observation_noise
    S = H @ ensemble_covariance @ H.T + R
    K = ensemble_covariance @ H.T @ np.linalg.inv(S)

    # Update ensemble members
    updated_ensemble = np.zeros_like(ensemble)
    for i, member in enumerate(ensemble):
        if not deterministic:
            # Stochastic update with perturbed observations
            perturbed_observation = observations + np.random.multivariate_normal(np.zeros(len(observations)), observation_noise)
            innovation = perturbed_observation - H @ member
        else:
            # Deterministic update with exact observations
            innovation = observations - H @ member
        updated_ensemble[i] = member + K @ innovation

    return updated_ensemble

def enkf_update_stochastic(predicted_states, predicted_covariances, observations, observation_covariance, observation_matrix, ensemble_size):
    """
    Perform stochastic updates in the Ensemble Kalman Filter (EnKF) to update the state estimate based on new observations.
    
    Parameters:
        predicted_states (np.array): Array of predicted state estimates for each ensemble member.
        predicted_covariances (np.array): Array of predicted covariance matrices for each ensemble member.
        observations (np.array): The new observation vector.
        observation_covariance (np.array): The covariance matrix of the observation error.
        observation_matrix (np.array): The matrix that maps the state space to the observation space.
        ensemble_size (int): Number of ensemble members.
    
    Returns:
        np.array: The updated ensemble of state estimates.
        np.array: The updated ensemble of covariance matrices.
    """
    # Initialize arrays to store updated states and covariances
    updated_states = np.zeros_like(predicted_states)
    updated_covariances = np.zeros_like(predicted_covariances)
    
    # Compute Kalman gain for each ensemble member
    for i in range(ensemble_size):
        # Compute Kalman gain
        kalman_gain = predicted_covariances[i] @ observation_matrix.T @ np.linalg.inv(observation_matrix @ predicted_covariances[i] @ observation_matrix.T + observation_covariance)
        
        # Update state estimate using stochastic update
        innovation = np.random.multivariate_normal(mean=np.zeros_like(observations), cov=observation_covariance)
        updated_states[i] = predicted_states[i] + kalman_gain @ (observations + innovation - observation_matrix @ predicted_states[i])
        
        # Update covariance matrix
        updated_covariances[i] = predicted_covariances[i] - kalman_gain @ observation_matrix @ predicted_covariances[i]
    
    return updated_states, updated_covariances

def enkf_update_deterministic(predicted_states, predicted_covariances, observations, observation_covariance, observation_matrix, ensemble_size):
    """
    Perform deterministic updates in the Ensemble Kalman Filter (EnKF) to update the state estimate based on new observations.
    
    Parameters:
        predicted_states (np.array): Array of predicted state estimates for each ensemble member.
        predicted_covariances (np.array): Array of predicted covariance matrices for each ensemble member.
        observations (np.array): The new observation vector.
        observation_covariance (np.array): The covariance matrix of the observation error.
        observation_matrix (np.array): The matrix that maps the state space to the observation space.
        ensemble_size (int): Number of ensemble members.
    
    Returns:
        np.array: The updated ensemble of state estimates.
        np.array: The updated ensemble of covariance matrices.
    """
    # Compute ensemble mean
    ensemble_mean = np.mean(predicted_states, axis=0)
    
    # Compute ensemble covariance
    ensemble_covariance = np.cov(predicted_states, rowvar=False, ddof=0)
    
    # Compute Kalman gain
    kalman_gain = ensemble_covariance @ observation_matrix.T @ np.linalg.inv(observation_matrix @ ensemble_covariance @ observation_matrix.T + observation_covariance)
    
    # Update state estimate for each ensemble member using deterministic update
    updated_states = predicted_states + kalman_gain @ (observations - observation_matrix @ predicted_states).T
    
    # Update covariance matrix for each ensemble member
    updated_covariances = predicted_covariances - np.einsum('ijk,ikl->ijl', kalman_gain @ observation_matrix, predicted_covariances)
    
    return updated_states, updated_covariances

def enkf_update_localization(predicted_states, observations, observation_matrix, observation_noise, ensemble_size, localization_radius):
    """
    Perform localized updates in the Ensemble Kalman Filter (EnKF) to update the state estimate based on new observations.
    
    Parameters:
        predicted_states (np.array): Array of predicted state estimates for each ensemble member.
        observations (np.array): The new observation vector.
        observation_matrix (np.array): The matrix that maps the state space to the observation space.
        observation_noise (np.array): The observation noise covariance matrix.
        ensemble_size (int): Number of ensemble members.
        localization_radius (int): Radius of localization in terms of grid points.
    
    Returns:
        np.array: The updated ensemble of state estimates.
    """
    # Compute ensemble mean and anomalies
    ensemble_mean = np.mean(predicted_states, axis=0)
    anomalies = predicted_states - ensemble_mean

    # Compute ensemble covariance with localization
    ensemble_covariance = np.zeros((len(ensemble_mean), len(ensemble_mean)))
    for i in range(len(ensemble_mean)):
        for j in range(len(ensemble_mean)):
            distance = np.abs(i - j)
            if distance <= localization_radius:
                weight = 1 - distance / localization_radius
            else:
                weight = 0
            ensemble_covariance[i, j] = weight * np.mean(anomalies[:, i] * anomalies[:, j])

    # Compute Kalman gain
    H = observation_matrix
    R = observation_noise
    S = H @ ensemble_covariance @ H.T + R
    K = ensemble_covariance @ H.T @ np.linalg.inv(S)
    
    # Update each ensemble member
    updated_states = np.zeros_like(predicted_states)
    for i in range(ensemble_size):
        innovation = observations - H @ predicted_states[i]
        updated_states[i] = predicted_states[i] + K @ innovation
    
    return updated_states

def enkf_inflation(predicted_states, inflation_factor):
    """
    Apply inflation to the ensemble of state estimates in the Ensemble Kalman Filter (EnKF).
    
    Parameters:
        predicted_states (np.array): Array of predicted state estimates for each ensemble member.
        inflation_factor (float): Factor by which to inflate the ensemble spread.
    
    Returns:
        np.array: The inflated ensemble of state estimates.
    """
    # Compute ensemble mean
    ensemble_mean = np.mean(predicted_states, axis=0)
    
    # Compute ensemble spread
    ensemble_spread = np.std(predicted_states, axis=0, ddof=1)
    
    # Apply inflation to each ensemble member
    inflated_states = ensemble_mean + inflation_factor * (predicted_states - ensemble_mean)
    
    return inflated_states

# 3DVar
def J_3dvar(x, xb, B_inv, y, H, R_inv):
    """
    Cost function for 3DVar.

    Parameters:
        x (np.array): State vector being optimized.
        xb (np.array): Background state vector.
        B_inv (np.array): Inverse of the background error covariance matrix.
        y (np.array): Observation vector.
        H (np.array): Observation operator matrix.
        R_inv (np.array): Inverse of the observation error covariance matrix.

    Returns:
        float: Value of the cost function.
    """
    dx = x - xb
    innovation = y - H @ x
    return 0.5 * (dx.T @ B_inv @ dx + innovation.T @ R_inv @ innovation)

def grad_J_3dvar(x, xb, B_inv, y, H, R_inv):
    """
    Gradient of the cost function for 3DVar.

    Parameters:
        x (np.array): State vector being optimized.
        xb (np.array): Background state vector.
        B_inv (np.array): Inverse of the background error covariance matrix.
        y (np.array): Observation vector.
        H (np.array): Observation operator matrix.
        R_inv (np.array): Inverse of the observation error covariance matrix.

    Returns:
        np.array: Gradient of the cost function.
    """
    dx = x - xb
    innovation = y - H @ x
    return B_inv @ dx - H.T @ R_inv @ innovation

def three_d_var(xb, B, y, H, R):
    """
    3D Variational Data Assimilation.

    Parameters:
        xb (np.array): Background state vector.
        B (np.array): Background error covariance matrix.
        y (np.array): Observation vector.
        H (np.array): Observation operator matrix.
        R (np.array): Observation error covariance matrix.

    Returns:
        np.array: Analysis state vector.
    """
    B_inv = np.linalg.inv(B)
    R_inv = np.linalg.inv(R)
    result = minimize(J_3dvar, xb, args=(xb, B_inv, y, H, R_inv), jac=grad_J_3dvar)
    return result.x

# 4DVar
def model_propagation(x0, model, times):
    """
    Propagate state x0 through the model over specified times.
    
    Parameters:
        x0 (np.array): Initial state vector.
        model (callable): Model function describing state evolution.
        times (list): List of times for state propagation.
        
    Returns:
        list: List of states over time.
    """
    states = [x0]
    for t in range(1, len(times)):
        # Simple model propagation: x_t = f(x_{t-1})
        states.append(model(states[-1]))
    return states

def J_4dvar(x0, xb, B_inv, y, H, R_inv, model, times):
    """
    4DVar cost function.
    
    Parameters:
        x0 (np.array): Initial state vector being optimized.
        xb (np.array): Background state vector.
        B_inv (np.array): Inverse of the background error covariance matrix.
        y (list): List of observation vectors.
        H (list): List of observation operator matrices.
        R_inv (list): List of inverse observation error covariance matrices.
        model (callable): Model function describing state evolution.
        times (list): List of times for state propagation.
        
    Returns:
        float: Value of the cost function.
    """
    states = model_propagation(x0, model, times)
    cost = 0.5 * np.dot(x0 - xb, B_inv @ (x0 - xb))
    for k, obs in enumerate(y):
        innovation = obs - H[k] @ states[k]
        cost += 0.5 * np.dot(innovation, R_inv[k] @ innovation)
    return cost

def four_d_var(xb, B, y, H, R, model, times):
    '''
    4Dvar data assimilation
    
    Parameters:
        xb (np.array): Background state vector.
        B (np.array): Background error covariance matrix.
        y (list): List of observation vectors.
        H (list): List of observation operator matrices.
        R (list): List of observation error covariance matrices.
        model (callable): Model function describing state evolution.
        times (list): List of times for state propagation.
        
    Returns:
        np.array: Analysis state vector.
    '''
    B_inv = np.linalg.inv(B)
    R_inv = [np.linalg.inv(r) for r in R]
    result = minimize(J_4dvar, xb, args=(xb, B_inv, y, H, R_inv, model, times), method='L-BFGS-B')
    return result.x