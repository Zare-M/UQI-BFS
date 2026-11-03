# ============================================================
# LIBRARIES
# ============================================================

library(torch)
library(randomForest)
library(kernlab)
library(glmnet)
library(pROC)
library(rpart)
library(xgboost)


# ============================================================
# MAIN FUNCTION
# ============================================================

evaluate_uqi_feature_selection_binary_fixed <- function(
    X = X,
    Y = Y,
    top_k = 50,
    n_iter = 200,
    lr = 0.05,
    epsilon = 0.01,
    external_fraction = 0.20,
    n_folds = 5,
    n_repeats = 20,
    seed = 123,
    device = "cpu",
    verbose = FALSE
) {
  
  
  # ==========================================================
  # 1. BASIC VALIDATION
  # ==========================================================
  
  if (!is.matrix(X) && !is.data.frame(X)) {
    stop("X must be a matrix or data.frame.")
  }
  
  if (!is.data.frame(Y) && !is.matrix(Y)) {
    stop("Y must be a data.frame or matrix.")
  }
  
  if (top_k < 1) {
    stop("top_k must be >= 1.")
  }
  
  if (n_iter < 1) {
    stop("n_iter must be >= 1.")
  }
  
  if (lr <= 0) {
    stop("lr must be > 0.")
  }
  
  if (epsilon <= 0 || epsilon >= 1) {
    stop("epsilon must be > 0 and < 1.")
  }
  
  if (
    external_fraction <= 0 ||
    external_fraction >= 1
  ) {
    stop(
      "external_fraction must be > 0 and < 1."
    )
  }
  
  if (n_folds < 2) {
    stop("n_folds must be >= 2.")
  }
  
  if (n_repeats < 1) {
    stop("n_repeats must be >= 1.")
  }
  
  
  # ==========================================================
  # OUTER REPEAT LOOP
  # ==========================================================
  
  all_cv_tables <- vector("list", n_repeats)
  all_external_tables <- vector("list", n_repeats)
  
  last_result <- NULL
  
  for (rep in seq_len(n_repeats)) {
    
    current_seed <- seed + (rep - 1) * 1000
    
    set.seed(current_seed)
    
    if (isTRUE(verbose)) {
      cat("\n============================================================\n")
      cat("REPEAT:", rep, "/", n_repeats, " | seed:", current_seed, "\n")
      cat("============================================================\n")
    }
    
    
    # ==========================================================
    # 2. PREPARE X
    # ==========================================================
    
    if (ncol(X) < 2) {
      stop(
        "X must contain at least two columns: sample ID + features."
      )
    }
    
    X_sample_ids <- as.character(
      X[[1]]
    )
    
    if (
      anyDuplicated(X_sample_ids)
    ) {
      stop(
        "Sample IDs in the first column of X must be unique."
      )
    }
    
    X_features <- X[
      ,
      -1,
      drop = FALSE
    ]
    
    X_features <- as.matrix(
      X_features
    )
    
    suppressWarnings(
      storage.mode(X_features) <- "numeric"
    )
    
    
    # ==========================================================
    # 3. PREPARE Y
    # ==========================================================
    
    Y <- as.data.frame(
      Y,
      stringsAsFactors = FALSE
    )
    
    required_columns <- c(
      "sample",
      "class",
      "group_id"
    )
    
    missing_columns <- setdiff(
      required_columns,
      colnames(Y)
    )
    
    if (
      length(missing_columns) > 0
    ) {
      stop(
        paste0(
          "Y is missing required columns: ",
          paste(
            missing_columns,
            collapse = ", "
          )
        )
      )
    }
    
    Y$sample <- as.character(
      Y$sample
    )
    
    Y$group_id <- as.character(
      Y$group_id
    )
    
    
    if (
      anyDuplicated(Y$sample)
    ) {
      stop(
        "The 'sample' column in Y must be unique."
      )
    }
    
    
    # ==========================================================
    # 4. ALIGN X AND Y USING SAMPLE ID
    # ==========================================================
    
    match_idx <- match(
      X_sample_ids,
      Y$sample
    )
    
    if (
      anyNA(match_idx)
    ) {
      
      missing_samples <- X_sample_ids[
        is.na(match_idx)
      ]
      
      stop(
        paste0(
          length(missing_samples),
          " samples from X were not found in Y. ",
          "Example: ",
          paste(
            head(missing_samples, 5),
            collapse = ", "
          )
        )
      )
    }
    
    Y <- Y[
      match_idx,
      ,
      drop = FALSE
    ]
    
    if (
      !identical(
        X_sample_ids,
        Y$sample
      )
    ) {
      stop(
        "X and Y could not be aligned correctly by sample ID."
      )
    }
    
    
    # ==========================================================
    # 5. CREATE INTERNAL X / Y / GROUP OBJECTS
    # ==========================================================
    
    X_mat <- X_features
    
    feature_names <- colnames(X_mat)
    
    if (
      is.null(feature_names) ||
      length(feature_names) != ncol(X_mat) ||
      all(is.na(feature_names)) ||
      all(feature_names == "")
    ) {
      
      feature_names <- paste0(
        "Feature_",
        seq_len(ncol(X_mat))
      )
      
      colnames(X_mat) <- feature_names
      
    } else {
      
      feature_names <- make.unique(
        feature_names
      )
      
      colnames(X_mat) <- feature_names
    }
    
    
    # ==========================================================
    # 6. CONVERT Y TO BINARY
    # ==========================================================
    
    Y_raw <- Y$class
    
    if (is.factor(Y_raw)) {
      
      original_levels <- levels(
        Y_raw
      )
      
      if (
        length(original_levels) != 2
      ) {
        stop(
          "Y$class must contain exactly two classes."
        )
      }
      
      y_numeric <- as.numeric(
        Y_raw
      ) - 1
      
    } else if (is.character(Y_raw)) {
      
      valid_values <- unique(
        Y_raw[
          !is.na(Y_raw) &
            Y_raw != ""
        ]
      )
      
      if (
        length(valid_values) != 2
      ) {
        stop(
          "Character Y$class must contain exactly two classes."
        )
      }
      
      valid_values <- sort(
        valid_values
      )
      
      original_levels <- valid_values
      
      y_numeric <- ifelse(
        Y_raw == valid_values[1],
        0,
        ifelse(
          Y_raw == valid_values[2],
          1,
          NA_real_
        )
      )
      
    } else {
      
      suppressWarnings(
        Y_numeric_raw <- as.numeric(
          Y_raw
        )
      )
      
      unique_y <- sort(
        unique(
          Y_numeric_raw[
            is.finite(Y_numeric_raw)
          ]
        )
      )
      
      if (
        length(unique_y) != 2
      ) {
        stop(
          "Numeric Y$class must contain exactly two unique finite classes."
        )
      }
      
      original_levels <- as.character(
        unique_y
      )
      
      y_numeric <- ifelse(
        Y_numeric_raw == unique_y[1],
        0,
        ifelse(
          Y_numeric_raw == unique_y[2],
          1,
          NA_real_
        )
      )
    }
    
    y_numeric <- as.numeric(
      y_numeric
    )
    
    
    # ==========================================================
    # 7. GROUP IDs
    # ==========================================================
    
    group_id <- as.character(
      Y$group_id
    )
    
    if (
      any(
        is.na(group_id) |
        group_id == ""
      )
    ) {
      stop(
        "group_id contains NA or empty values."
      )
    }
    
    
    # ==========================================================
    # 8. DIMENSION CHECK
    # ==========================================================
    
    if (
      nrow(X_mat) != length(y_numeric)
    ) {
      stop(
        paste0(
          "Number of rows in X (",
          nrow(X_mat),
          ") does not match Y (",
          length(y_numeric),
          ")."
        )
      )
    }
    
    
    if (
      length(group_id) != nrow(X_mat)
    ) {
      stop(
        "Length of group_id does not match number of X samples."
      )
    }
    
    
    # ==========================================================
    # 9. REMOVE INVALID Y SAMPLES
    # ==========================================================
    
    valid_y <- is.finite(
      y_numeric
    )
    
    X_mat <- X_mat[
      valid_y,
      ,
      drop = FALSE
    ]
    
    y_numeric <- y_numeric[
      valid_y
    ]
    
    group_id <- group_id[
      valid_y
    ]
    
    sample_ids <- X_sample_ids[
      valid_y
    ]
    
    
    # ==========================================================
    # 10. VERIFY FINAL CLASSES
    # ==========================================================
    
    if (
      length(
        unique(y_numeric)
      ) != 2
    ) {
      stop(
        "After removing invalid Y values, exactly two classes must remain."
      )
    }
    
    if (
      any(
        table(y_numeric) < 2
      )
    ) {
      stop(
        "Each class must contain at least two samples."
      )
    }
    
    
    # ==========================================================
    # 11. GROUP INFORMATION
    # ==========================================================
    
    group_table <- table(
      group_id
    )
    
    unique_groups <- unique(
      group_id
    )
    
    n_groups <- length(
      unique_groups
    )
    
    if (
      n_groups < n_folds
    ) {
      stop(
        paste0(
          "Number of unique groups (",
          n_groups,
          ") is smaller than n_folds (",
          n_folds,
          ")."
        )
      )
    }
    
    
    # ==========================================================
    # 12. GROUP-WISE EXTERNAL SPLIT
    # ==========================================================
    
    make_group_external_split <- function(
    y,
    groups,
    external_fraction,
    seed = 123,
    n_attempts = 1000
    ) {
      
      set.seed(seed)
      
      unique_groups <- unique(
        groups
      )
      
      group_stats <- data.frame(
        group = unique_groups,
        n = integer(
          length(unique_groups)
        ),
        n0 = integer(
          length(unique_groups)
        ),
        n1 = integer(
          length(unique_groups)
        ),
        stringsAsFactors = FALSE
      )
      
      for (
        i in seq_along(unique_groups)
      ) {
        
        idx <- which(
          groups ==
            unique_groups[i]
        )
        
        group_stats$n[i] <-
          length(idx)
        
        group_stats$n0[i] <-
          sum(y[idx] == 0)
        
        group_stats$n1[i] <-
          sum(y[idx] == 1)
      }
      
      n_total <- length(y)
      
      target_total <-
        external_fraction *
        n_total
      
      target_0 <-
        external_fraction *
        sum(y == 0)
      
      target_1 <-
        external_fraction *
        sum(y == 1)
      
      best_score <- Inf
      best_external_groups <- NULL
      
      for (
        attempt in seq_len(n_attempts)
      ) {
        
        order_idx <- order(
          runif(nrow(group_stats)),
          -group_stats$n
        )
        
        assigned <- character(0)
        
        current_total <- 0
        current_0 <- 0
        current_1 <- 0
        
        for (
          ii in order_idx
        ) {
          
          g_n <- group_stats$n[ii]
          g_0 <- group_stats$n0[ii]
          g_1 <- group_stats$n1[ii]
          
          current_score <-
            abs(
              current_total -
                target_total
            ) /
            max(
              target_total,
              1
            ) +
            abs(
              current_0 -
                target_0
            ) /
            max(
              target_0,
              1
            ) +
            abs(
              current_1 -
                target_1
            ) /
            max(
              target_1,
              1
            )
          
          proposed_score <-
            abs(
              (current_total + g_n) -
                target_total
            ) /
            max(
              target_total,
              1
            ) +
            abs(
              (current_0 + g_0) -
                target_0
            ) /
            max(
              target_0,
              1
            ) +
            abs(
              (current_1 + g_1) -
                target_1
            ) /
            max(
              target_1,
              1
            )
          
          if (
            proposed_score < current_score ||
            current_total < target_total * 0.8
          ) {
            
            assigned <- c(
              assigned,
              unique_groups[ii]
            )
            
            current_total <-
              current_total + g_n
            
            current_0 <-
              current_0 + g_0
            
            current_1 <-
              current_1 + g_1
          }
        }
        
        external_idx <- which(
          groups %in% assigned
        )
        
        training_idx <- setdiff(
          seq_along(y),
          external_idx
        )
        
        if (
          length(
            unique(y[external_idx])
          ) != 2 ||
          length(
            unique(y[training_idx])
          ) != 2
        ) {
          next
        }
        
        actual_total <-
          length(external_idx)
        
        actual_0 <-
          sum(y[external_idx] == 0)
        
        actual_1 <-
          sum(y[external_idx] == 1)
        
        score <-
          abs(
            actual_total -
              target_total
          ) /
          max(
            target_total,
            1
          ) +
          abs(
            actual_0 -
              target_0
          ) /
          max(
            target_0,
            1
          ) +
          abs(
            actual_1 -
              target_1
          ) /
          max(
            target_1,
            1
          )
        
        if (
          score < best_score
        ) {
          
          best_score <- score
          
          best_external_groups <-
            assigned
        }
      }
      
      if (
        is.null(
          best_external_groups
        )
      ) {
        stop(
          "Could not construct a valid group-wise external split."
        )
      }
      
      external_idx <- which(
        groups %in%
          best_external_groups
      )
      
      training_idx <- setdiff(
        seq_along(y),
        external_idx
      )
      
      train_groups <- unique(
        groups[training_idx]
      )
      
      test_groups <- unique(
        groups[external_idx]
      )
      
      overlap <- intersect(
        train_groups,
        test_groups
      )
      
      if (
        length(overlap) > 0
      ) {
        stop(
          "GROUP LEAKAGE DETECTED in external split."
        )
      }
      
      list(
        training_indices = training_idx,
        external_indices = external_idx,
        training_groups = train_groups,
        external_groups = test_groups
      )
    }
    
    
    external_split <-
      make_group_external_split(
        y = y_numeric,
        groups = group_id,
        external_fraction =
          external_fraction,
        seed = current_seed
      )
    
    training_indices <-
      external_split$training_indices
    
    external_indices <-
      external_split$external_indices
    
    X_training <- X_mat[
      training_indices,
      ,
      drop = FALSE
    ]
    
    y_training <- y_numeric[
      training_indices
    ]
    
    group_training <- group_id[
      training_indices
    ]
    
    X_external <- X_mat[
      external_indices,
      ,
      drop = FALSE
    ]
    
    y_external <- y_numeric[
      external_indices
    ]
    
    group_external <- group_id[
      external_indices
    ]
    
    
    # ==========================================================
    # 13. EXTERNAL GROUP LEAKAGE AUDIT
    # ==========================================================
    
    if (
      length(
        intersect(
          unique(group_training),
          unique(group_external)
        )
      ) > 0
    ) {
      stop(
        "GROUP LEAKAGE DETECTED between training and external test."
      )
    }
    
    
    # ==========================================================
    # 14. GROUP-WISE STRATIFIED K-FOLD
    # ==========================================================
    
    make_group_stratified_folds <- function(
    y,
    groups,
    n_folds = 5,
    seed = 123,
    n_attempts = 500
    ) {
      
      if (
        length(y) != length(groups)
      ) {
        stop(
          "y and groups must have the same length."
        )
      }
      
      if (
        length(unique(y)) != 2
      ) {
        stop(
          "Exactly two classes are required."
        )
      }
      
      unique_groups <- unique(
        groups
      )
      
      if (
        length(unique_groups) <
        n_folds
      ) {
        stop(
          "Number of groups is smaller than number of folds."
        )
      }
      
      group_stats <- data.frame(
        group = unique_groups,
        n = 0,
        n0 = 0,
        n1 = 0,
        stringsAsFactors = FALSE
      )
      
      for (
        i in seq_along(unique_groups)
      ) {
        
        idx <- which(
          groups ==
            unique_groups[i]
        )
        
        group_stats$n[i] <-
          length(idx)
        
        group_stats$n0[i] <-
          sum(y[idx] == 0)
        
        group_stats$n1[i] <-
          sum(y[idx] == 1)
      }
      
      target_n <-
        nrow(group_stats) /
        n_folds
      
      target_n0 <-
        sum(y == 0) /
        n_folds
      
      target_n1 <-
        sum(y == 1) /
        n_folds
      
      best_assignment <- NULL
      best_score <- Inf
      
      set.seed(seed)
      
      for (
        attempt in seq_len(n_attempts)
      ) {
        
        fold_n <- numeric(
          n_folds
        )
        
        fold_n0 <- numeric(
          n_folds
        )
        
        fold_n1 <- numeric(
          n_folds
        )
        
        assignment <- integer(
          nrow(group_stats)
        )
        
        mixed <- (
          group_stats$n0 > 0 &
            group_stats$n1 > 0
        )
        
        ord <- order(
          !mixed,
          -group_stats$n,
          runif(
            nrow(group_stats)
          )
        )
        
        for (
          ii in ord
        ) {
          
          scores <- numeric(
            n_folds
          )
          
          for (
            f in seq_len(n_folds)
          ) {
            
            new_n <-
              fold_n[f] +
              group_stats$n[ii]
            
            new_n0 <-
              fold_n0[f] +
              group_stats$n0[ii]
            
            new_n1 <-
              fold_n1[f] +
              group_stats$n1[ii]
            
            scores[f] <-
              
              abs(
                new_n -
                  target_n
              ) /
              max(
                target_n,
                1
              ) +
              
              abs(
                new_n0 -
                  target_n0
              ) /
              max(
                target_n0,
                1
              ) +
              
              abs(
                new_n1 -
                  target_n1
              ) /
              max(
                target_n1,
                1
              )
          }
          
          min_score <- min(
            scores
          )
          
          candidate_folds <- which(
            abs(
              scores -
                min_score
            ) < 1e-12
          )
          
          chosen_fold <- sample(
            candidate_folds,
            1
          )
          
          assignment[ii] <-
            chosen_fold
          
          fold_n[chosen_fold] <-
            fold_n[chosen_fold] +
            group_stats$n[ii]
          
          fold_n0[chosen_fold] <-
            fold_n0[chosen_fold] +
            group_stats$n0[ii]
          
          fold_n1[chosen_fold] <-
            fold_n1[chosen_fold] +
            group_stats$n1[ii]
        }
        
        valid <- all(
          fold_n0 > 0 &
            fold_n1 > 0
        )
        
        if (!valid) {
          next
        }
        
        score <-
          
          sum(
            abs(
              fold_n -
                target_n
            ) /
              max(
                target_n,
                1
              )
          ) +
          
          sum(
            abs(
              fold_n0 -
                target_n0
            ) /
              max(
                target_n0,
                1
              )
          ) +
          
          sum(
            abs(
              fold_n1 -
                target_n1
            ) /
              max(
                target_n1,
                1
              )
          )
        
        if (
          score < best_score
        ) {
          
          best_score <- score
          
          best_assignment <-
            assignment
        }
      }
      
      if (
        is.null(
          best_assignment
        )
      ) {
        stop(
          "Could not construct valid group-wise stratified folds."
        )
      }
      
      fold_id <- integer(
        length(y)
      )
      
      for (
        i in seq_along(unique_groups)
      ) {
        
        fold_id[
          groups ==
            unique_groups[i]
        ] <-
          best_assignment[i]
      }
      
      for (
        f in seq_len(n_folds)
      ) {
        
        idx <- which(
          fold_id == f
        )
        
        if (
          length(idx) == 0
        ) {
          stop(
            paste(
              "Fold",
              f,
              "is empty."
            )
          )
        }
        
        if (
          length(
            unique(y[idx])
          ) != 2
        ) {
          stop(
            paste(
              "Fold",
              f,
              "does not contain both classes."
            )
          )
        }
      }
      
      group_fold_count <- sapply(
        unique_groups,
        function(g) {
          
          length(
            unique(
              fold_id[
                groups == g
              ]
            )
          )
        }
      )
      
      if (
        any(
          group_fold_count != 1
        )
      ) {
        stop(
          "GROUP LEAKAGE DETECTED inside cross-validation folds."
        )
      }
      
      fold_id
    }
    
    
    # ==========================================================
    # 15. TRAINING FOLDS
    # ==========================================================
    
    fold_id <- make_group_stratified_folds(
      y = y_training,
      groups = group_training,
      n_folds = n_folds,
      seed = current_seed
    )
    
    
    # ==========================================================
    # 16. IMPUTATION HELPERS
    # ==========================================================
    
    compute_column_medians <- function(
    X_ref
    ) {
      
      medians <- apply(
        X_ref,
        2,
        function(col) {
          
          valid <- is.finite(
            col
          )
          
          if (
            any(valid)
          ) {
            
            med <- median(
              col[valid],
              na.rm = TRUE
            )
            
          } else {
            
            med <- 0
          }
          
          if (
            !is.finite(med)
          ) {
            med <- 0
          }
          
          med
        }
      )
      
      medians
    }
    
    
    apply_median_imputation <- function(
    X_target,
    medians
    ) {
      
      for (
        j in seq_len(
          ncol(X_target)
        )
      ) {
        
        invalid <- !is.finite(
          X_target[, j]
        )
        
        if (
          any(invalid)
        ) {
          
          X_target[
            invalid,
            j
          ] <- medians[j]
        }
      }
      
      X_target
    }
    
    
    # ==========================================================
    # 17. CLASS COUNTS
    # ==========================================================
    
    class_counts <- table(
      y_numeric
    )
    
    
    # ==========================================================
    # 18. UQI FUNCTION
    # ==========================================================
    
    uqi_torch <- function(
    x,
    y,
    offdiag_mask,
    eps = 1e-8
    ) {
      
      x_values <- torch_masked_select(
        x,
        offdiag_mask
      )
      
      y_values <- torch_masked_select(
        y,
        offdiag_mask
      )
      
      if (
        x_values$numel() == 0
      ) {
        stop(
          "No off-diagonal elements available for UQI."
        )
      }
      
      mu_x <- torch_mean(
        x_values
      )
      
      mu_y <- torch_mean(
        y_values
      )
      
      var_x <- torch_mean(
        (
          x_values -
            mu_x
        )^2
      )
      
      var_y <- torch_mean(
        (
          y_values -
            mu_y
        )^2
      )
      
      cov_xy <- torch_mean(
        (
          x_values -
            mu_x
        ) *
          (
            y_values -
              mu_y
          )
      )
      
      numerator <-
        4 *
        mu_x *
        mu_y *
        cov_xy
      
      denominator <-
        (
          mu_x^2 +
            mu_y^2 +
            eps
        ) *
        (
          var_x +
            var_y +
            eps
        )
      
      numerator /
        denominator
    }
    
    
    # ==========================================================
    # 19. TARGET SIMILARITY MATRIX T
    # ==========================================================
    
    create_target_similarity <- function(
    y
    ) {
      
      n <- length(y)
      
      target <- matrix(
        epsilon,
        nrow = n,
        ncol = n
      )
      
      same_class <- outer(
        y,
        y,
        "=="
      )
      
      target[
        same_class
      ] <- 1
      
      diag(target) <- 1
      
      target
    }
    
    
    # ==========================================================
    # 20. SIMILARITY CHANNELS
    # ==========================================================
    
    calculate_similarity_channels <- function(
    X_scaled,
    gaussian_sigma
    ) {
      
      X_scaled <- as.matrix(
        X_scaled
      )
      
      X_tensor <- torch_tensor(
        X_scaled,
        dtype = torch_float(),
        device = device
      )
      
      
      # --------------------------------------------------------
      # Pairwise Euclidean distances
      # --------------------------------------------------------
      
      gram <- torch_matmul(
        X_tensor,
        X_tensor$t()
      )
      
      sq_norms <-
        torch_sum(
          X_tensor^2,
          dim = 2,
          keepdim = TRUE
        )
      
      dist_sq <-
        sq_norms +
        sq_norms$t() -
        2 * gram
      
      dist_sq <- torch_clamp(
        dist_sq,
        min = 0
      )
      
      
      # --------------------------------------------------------
      # Euclidean similarity
      # --------------------------------------------------------
      
      euclidean_distance <-
        torch_sqrt(
          dist_sq +
            1e-8
        )
      
      euclidean_similarity <-
        1 /
        (
          1 +
            euclidean_distance
        )
      
      
      # --------------------------------------------------------
      # Gaussian similarity
      # --------------------------------------------------------
      
      gaussian_similarity <-
        torch_exp(
          -dist_sq /
            (
              2 *
                gaussian_sigma^2 +
                1e-8
            )
        )
      
      
      # --------------------------------------------------------
      # Pearson similarity
      # --------------------------------------------------------
      
      X_centered <-
        X_tensor -
        torch_mean(
          X_tensor,
          dim = 2,
          keepdim = TRUE
        )
      
      numerator <-
        torch_matmul(
          X_centered,
          X_centered$t()
        )
      
      norm <-
        torch_sqrt(
          torch_sum(
            X_centered^2,
            dim = 2,
            keepdim = TRUE
          ) +
            1e-8
        )
      
      denominator <-
        torch_matmul(
          norm,
          norm$t()
        ) +
        1e-8
      
      pearson_similarity <-
        numerator /
        denominator
      
      pearson_similarity <-
        torch_clamp(
          pearson_similarity,
          min = -1,
          max = 1
        )
      
      # Map Pearson to [0, 1]
      pearson_similarity <- (pearson_similarity + 1) / 2
      
      
      list(
        Pearson =
          pearson_similarity,
        Gaussian =
          gaussian_similarity,
        Euclidean =
          euclidean_similarity
      )
    }
    
    
    # ==========================================================
    # 21. UQI FEATURE SELECTION
    # ==========================================================
    
    select_features_uqi <- function(
    X_train,
    y_train,
    top_k = 100,
    n_iter = 20,
    lr = 0.05,
    epsilon = 0.01,
    device = "cpu",
    verbose = FALSE,
    uqi_seed = 123
    ) {
      
      
      # --------------------------------------------------------
      # Remove zero-variance features
      # --------------------------------------------------------
      
      feature_var <- apply(
        X_train,
        2,
        var,
        na.rm = TRUE
      )
      
      valid_features <-
        is.finite(feature_var) &
        feature_var > 1e-12
      
      X_train <- X_train[
        ,
        valid_features,
        drop = FALSE
      ]
      
      current_feature_names <-
        colnames(
          X_train
        )
      
      if (
        ncol(X_train) == 0
      ) {
        stop(
          "No valid features remain for UQI."
        )
      }
      
      top_k_actual <- min(
        top_k,
        ncol(X_train)
      )
      
      
      # --------------------------------------------------------
      # FULL REPRESENTATION
      # --------------------------------------------------------
      
      centers <- colMeans(
        X_train,
        na.rm = TRUE
      )
      
      scales <- apply(
        X_train,
        2,
        sd,
        na.rm = TRUE
      )
      
      scales[
        !is.finite(scales) |
          scales == 0
      ] <- 1
      
      X_full_scaled <- sweep(
        X_train,
        2,
        centers,
        "-"
      )
      
      X_full_scaled <- sweep(
        X_full_scaled,
        2,
        scales,
        "/"
      )
      
      X_full_scaled[
        !is.finite(X_full_scaled)
      ] <- 0
      
      
      # --------------------------------------------------------
      # TARGET MATRIX
      # --------------------------------------------------------
      
      T_matrix <-
        create_target_similarity(
          y_train
        )
      
      
      # --------------------------------------------------------
      # OFF-DIAGONAL MASK
      # --------------------------------------------------------
      
      n <- nrow(
        X_full_scaled
      )
      
      if (
        n < 2
      ) {
        stop(
          "At least two samples are required for UQI."
        )
      }
      
      offdiag_mask_R <- matrix(
        TRUE,
        nrow = n,
        ncol = n
      )
      
      diag(
        offdiag_mask_R
      ) <- FALSE
      
      
      # --------------------------------------------------------
      # GAUSSIAN SIGMA
      # --------------------------------------------------------
      
      pairwise_distances <- as.vector(
        dist(
          X_full_scaled
        )
      )
      
      valid_distances <-
        pairwise_distances[
          is.finite(
            pairwise_distances
          ) &
            pairwise_distances > 0
        ]
      
      if (
        length(
          valid_distances
        ) > 0
      ) {
        
        gaussian_sigma <- median(
          valid_distances
        )
        
      } else {
        
        gaussian_sigma <- 1
      }
      
      if (
        !is.finite(
          gaussian_sigma
        ) ||
        gaussian_sigma <= 0
      ) {
        gaussian_sigma <- 1
      }
      
      
      # ========================================================
      # FULL SIMILARITY CHANNELS
      # ========================================================
      
      full_channels <-
        calculate_similarity_channels(
          X_full_scaled,
          gaussian_sigma
        )
      
      
      # ========================================================
      # REFERENCE CHANNELS (FULL FEATURE SPACE; NO CLASS MASK)
      # ========================================================
      # These matrices remain unmodified and serve as references.
      reference_pearson <- full_channels$Pearson
      reference_gaussian <- full_channels$Gaussian
      reference_euclidean <- full_channels$Euclidean
      
      # Target/class similarity matrix. It will be applied ONLY
      # to the selected-feature similarity matrices below.
      T_tensor <- torch_tensor(
        T_matrix,
        dtype = torch_float(),
        device = device
      )
      
      
      # --------------------------------------------------------
      # Full data tensor
      # --------------------------------------------------------
      
      X_full_tensor <- torch_tensor(
        X_full_scaled,
        dtype = torch_float(),
        device = device
      )
      
      offdiag_mask <- torch_tensor(
        offdiag_mask_R,
        dtype = torch_bool(),
        device = device
      )
      
      
      # ========================================================
      # LEARNABLE WEIGHTS
      # ========================================================
      
      torch_manual_seed(
        uqi_seed
      )
      
      w <- torch_randn(
        ncol(X_full_scaled),
        dtype = torch_float(),
        device = device
      ) * 0.01
      
      w$requires_grad_(
        TRUE
      )
      
      optimizer <- optim_adam(
        params = list(w),
        lr = lr
      )
      
      
      # ========================================================
      # HISTORY
      # ========================================================
      
      uqi_history <- numeric(
        n_iter
      )
      
      uqi_pearson_history <- numeric(
        n_iter
      )
      
      uqi_gaussian_history <- numeric(
        n_iter
      )
      
      uqi_euclidean_history <- numeric(
        n_iter
      )
      
      loss_history <- numeric(
        n_iter
      )
      
      grad_history <- numeric(
        n_iter
      )
      
      
      # ========================================================
      # OPTIMIZATION
      # ========================================================
      
      for (
        iter in seq_len(
          n_iter
        )
      ) {
        
        optimizer$zero_grad()
        
        
        # ------------------------------------------------------
        # Positive soft weights
        # ------------------------------------------------------
        
        positive_weights <-
          torch_sigmoid(
            w
          )
        
        
        # ------------------------------------------------------
        # HARD TOP-K
        # ------------------------------------------------------
        
        weight_values_raw <- as.numeric(
          w$detach()$cpu()
        )
        
        selected_idx <- order(
          weight_values_raw,
          decreasing = TRUE
        )[
          seq_len(
            top_k_actual
          )
        ]
        
        mask <- rep(
          0,
          length(
            weight_values_raw
          )
        )
        
        mask[
          selected_idx
        ] <- 1
        
        mask_tensor <- torch_tensor(
          mask,
          dtype = torch_float(),
          device = device
        )
        
        
        # ------------------------------------------------------
        # STRAIGHT-THROUGH ESTIMATOR
        # ------------------------------------------------------
        
        mask_ste <-
          mask_tensor +
          positive_weights -
          positive_weights$detach()
        
        effective_weights <-
          mask_ste
        
        
        # ======================================================
        # SELECTED REPRESENTATION
        # ======================================================
        
        X_selected <-
          X_full_tensor *
          effective_weights
        
        
        # ------------------------------------------------------
        # Selected Euclidean distance
        # ------------------------------------------------------
        
        gram <- torch_matmul(
          X_selected,
          X_selected$t()
        )
        
        sq_norms <-
          torch_sum(
            X_selected^2,
            dim = 2,
            keepdim = TRUE
          )
        
        dist_sq <-
          sq_norms +
          sq_norms$t() -
          2 * gram
        
        dist_sq <- torch_clamp(
          dist_sq,
          min = 0
        )
        
        
        # ------------------------------------------------------
        # Selected Euclidean similarity
        # ------------------------------------------------------
        
        euclidean_distance <-
          torch_sqrt(
            dist_sq +
              1e-8
          )
        
        selected_euclidean <-
          1 /
          (
            1 +
              euclidean_distance
          )
        
        
        # ------------------------------------------------------
        # Selected Gaussian similarity
        # ------------------------------------------------------
        
        selected_gaussian <-
          torch_exp(
            -dist_sq /
              (
                2 *
                  gaussian_sigma^2 +
                  1e-8
              )
          )
        
        
        # ------------------------------------------------------
        # Selected Pearson similarity: exact hard Top-k forward
        # with STE gradients
        # ------------------------------------------------------
        # Use the hard mask to compute means over exactly k selected
        # features. In the forward pass mask_ste equals mask_tensor,
        # so this is Pearson computed on the actual selected columns,
        # not Pearson on a full-width matrix with zeroed columns.
        # In the backward pass, mask_ste supplies the STE gradient.
        
        selected_sum <- torch_sum(
          X_full_tensor * effective_weights,
          dim = 2,
          keepdim = TRUE
        )
        
        selected_mean <- selected_sum / top_k_actual
        
        X_centered_full <-
          X_full_tensor - selected_mean
        
        weighted_centered <-
          X_centered_full * effective_weights
        
        numerator <- torch_matmul(
          weighted_centered,
          X_centered_full$t()
        )
        
        norm <- torch_sqrt(
          torch_sum(
            (X_centered_full^2) * effective_weights,
            dim = 2,
            keepdim = TRUE
          ) + 1e-8
        )
        
        denominator <- torch_matmul(
          norm,
          norm$t()
        ) + 1e-8
        
        selected_pearson <- numerator / denominator
        
        selected_pearson <- torch_clamp(
          selected_pearson,
          min = -1,
          max = 1
        )
        
        # Map Pearson from [-1, 1] to [0, 1]
        selected_pearson <- (selected_pearson + 1) / 2
        
        
        # ======================================================
        # APPLY T ONLY TO SELECTED-FEATURE CHANNELS
        # ======================================================
        selected_pearson_class <- selected_pearson * T_tensor
        selected_gaussian_class <- selected_gaussian * T_tensor
        selected_euclidean_class <- selected_euclidean * T_tensor
        
        
        # ======================================================
        # UQI: UNMASKED FULL-SPACE REFERENCE VS CLASS-GUIDED
        # SELECTED-FEATURE SIMILARITIES
        # ======================================================
        
        uqi_pearson <- uqi_torch(
          reference_pearson,
          selected_pearson_class,
          offdiag_mask
        )
        
        uqi_gaussian <- uqi_torch(
          reference_gaussian,
          selected_gaussian_class,
          offdiag_mask
        )
        
        uqi_euclidean <- uqi_torch(
          reference_euclidean,
          selected_euclidean_class,
          offdiag_mask
        )
        
        
        # ======================================================
        # MEAN UQI
        # ======================================================
        
        uqi_mean <-
          (
            uqi_pearson +
              uqi_gaussian +
              uqi_euclidean
          ) / 3
        
        
        # ======================================================
        # LOSS
        # ======================================================
        
        loss <-
          1 -
          uqi_mean
        
        
        # ======================================================
        # BACKPROPAGATION
        # ======================================================
        
        loss$backward()
        
        
        # ------------------------------------------------------
        # Gradient
        # ------------------------------------------------------
        
        if (
          is.null(w$grad)
        ) {
          grad_value <- NA_real_
        } else {
          
          grad_value <- as.numeric(
            torch_mean(
              torch_abs(
                w$grad
              )
            )$detach()$cpu()
          )
        }
        
        
        # ------------------------------------------------------
        # Optimizer
        # ------------------------------------------------------
        
        optimizer$step()
        
        
        # ======================================================
        # STORE HISTORY
        # ======================================================
        
        uqi_history[iter] <-
          as.numeric(
            uqi_mean$detach()$cpu()
          )
        
        uqi_pearson_history[iter] <-
          as.numeric(
            uqi_pearson$detach()$cpu()
          )
        
        uqi_gaussian_history[iter] <-
          as.numeric(
            uqi_gaussian$detach()$cpu()
          )
        
        uqi_euclidean_history[iter] <-
          as.numeric(
            uqi_euclidean$detach()$cpu()
          )
        
        loss_history[iter] <-
          as.numeric(
            loss$detach()$cpu()
          )
        
        grad_history[iter] <-
          grad_value
        
        
        if (
          isTRUE(verbose) &&
          (
            iter == 1 ||
            iter %% 10 == 0 ||
            iter == n_iter
          )
        ) {
          
          cat(
            "UQI iteration:",
            iter,
            "/",
            n_iter,
            " | UQI:",
            round(
              uqi_history[iter],
              6
            ),
            "\n"
          )
        }
      }
      
      
      # ========================================================
      # FINAL WEIGHTS
      # ========================================================
      
      final_weights <- as.numeric(
        torch_sigmoid(
          w
        )$detach()$cpu()
      )
      
      
      # ========================================================
      # FINAL TOP-K
      # ========================================================
      
      final_indices <- order(
        final_weights,
        decreasing = TRUE
      )[
        seq_len(
          top_k_actual
        )
      ]
      
      final_features <-
        current_feature_names[
          final_indices
        ]
      
      
      # ========================================================
      # RETURN
      # ========================================================
      
      list(
        
        selected_features =
          final_features,
        
        selected_indices =
          final_indices,
        
        weights =
          final_weights,
        
        target_similarity =
          T_matrix,
        
        reference = list(
          Pearson = reference_pearson,
          Gaussian = reference_gaussian,
          Euclidean = reference_euclidean
        ),
        
        reference_pearson =
          reference_pearson,
        
        reference_gaussian =
          reference_gaussian,
        
        reference_euclidean =
          reference_euclidean,
        
        uqi_history =
          uqi_history,
        
        uqi_pearson_history =
          uqi_pearson_history,
        
        uqi_gaussian_history =
          uqi_gaussian_history,
        
        uqi_euclidean_history =
          uqi_euclidean_history,
        
        loss_history =
          loss_history,
        
        grad_history =
          grad_history,
        
        gaussian_sigma_channel =
          gaussian_sigma,
        
        feature_names =
          current_feature_names
      )
    }
    
    
    # ==========================================================
    # 22. CLASSIFICATION METRICS
    # ==========================================================
    
    calculate_metrics <- function(
    y_true,
    y_prob
    ) {
      
      y_true <- as.numeric(
        y_true
      )
      
      y_prob <- as.numeric(
        y_prob
      )
      
      empty_metrics <- data.frame(
        Accuracy = NA_real_,
        Balanced_Accuracy = NA_real_,
        Sensitivity = NA_real_,
        Specificity = NA_real_,
        Precision_Class1 = NA_real_,
        Precision_Class0 = NA_real_,
        F1_Class1 = NA_real_,
        F1_Class0 = NA_real_,
        GMEAN = NA_real_,
        MCC = NA_real_,
        AUC = NA_real_
      )
      
      if (
        length(y_true) !=
        length(y_prob)
      ) {
        return(empty_metrics)
      }
      
      if (
        any(!is.finite(y_true)) ||
        any(!is.finite(y_prob))
      ) {
        return(empty_metrics)
      }
      
      y_prob <- pmin(
        pmax(
          y_prob,
          0
        ),
        1
      )
      
      y_pred <- ifelse(
        y_prob >= 0.5,
        1,
        0
      )
      
      TP <- sum(
        y_true == 1 &
          y_pred == 1
      )
      
      TN <- sum(
        y_true == 0 &
          y_pred == 0
      )
      
      FP <- sum(
        y_true == 0 &
          y_pred == 1
      )
      
      FN <- sum(
        y_true == 1 &
          y_pred == 0
      )
      
      total <-
        TP + TN + FP + FN
      
      accuracy <-
        if (total > 0)
          (TP + TN) / total
      else
        NA_real_
      
      sensitivity <-
        if ((TP + FN) > 0)
          TP / (TP + FN)
      else
        NA_real_
      
      specificity <-
        if ((TN + FP) > 0)
          TN / (TN + FP)
      else
        NA_real_
      
      precision_class1 <-
        if ((TP + FP) > 0)
          TP / (TP + FP)
      else
        NA_real_
      
      precision_class0 <-
        if ((TN + FN) > 0)
          TN / (TN + FN)
      else
        NA_real_
      
      f1_class1 <-
        if (
          is.finite(precision_class1) &&
          is.finite(sensitivity) &&
          (precision_class1 + sensitivity) > 0
        )
          2 * precision_class1 * sensitivity /
        (precision_class1 + sensitivity)
      else
        NA_real_
      
      f1_class0 <-
        if (
          is.finite(precision_class0) &&
          is.finite(specificity) &&
          (precision_class0 + specificity) > 0
        )
          2 * precision_class0 * specificity /
        (precision_class0 + specificity)
      else
        NA_real_
      
      gmean <-
        if (
          is.finite(sensitivity) &&
          is.finite(specificity) &&
          sensitivity >= 0 &&
          specificity >= 0
        )
          sqrt(sensitivity * specificity)
      else
        NA_real_
      
      balanced_accuracy <-
        if (
          is.finite(sensitivity) &&
          is.finite(specificity)
        )
          (sensitivity + specificity) / 2
      else
        NA_real_
      
      mcc_denominator <- sqrt(
        (TP + FP) *
          (TP + FN) *
          (TN + FP) *
          (TN + FN)
      )
      
      mcc_numerator <-
        TP * TN -
        FP * FN
      
      MCC <-
        if (
          is.finite(mcc_denominator) &&
          mcc_denominator > 0
        )
          mcc_numerator /
        mcc_denominator
      else
        NA_real_
      
      if (
        length(
          unique(y_true)
        ) == 2
      ) {
        
        AUC <- tryCatch(
          
          {
            
            roc_object <- pROC::roc(
              response = y_true,
              predictor = y_prob,
              quiet = TRUE,
              direction = "<"
            )
            
            as.numeric(
              pROC::auc(
                roc_object
              )
            )
          },
          
          error = function(e) {
            NA_real_
          }
        )
        
      } else {
        
        AUC <- NA_real_
      }
      
      data.frame(
        Accuracy = accuracy,
        Balanced_Accuracy = balanced_accuracy,
        Sensitivity = sensitivity,
        Specificity = specificity,
        Precision_Class1 = precision_class1,
        Precision_Class0 = precision_class0,
        F1_Class1 = f1_class1,
        F1_Class0 = f1_class0,
        GMEAN = gmean,
        MCC = MCC,
        AUC = AUC
      )
    }
    
    
    # ==========================================================
    # 23. MODEL FITTING
    # ==========================================================
    
    fit_model <- function(
    model_name,
    X_train,
    y_train,
    groups_train = NULL
    ) {
      
      y_factor <- factor(
        y_train,
        levels = c(0, 1)
      )
      
      
      # --------------------------------------------------------
      # Logistic Regression
      # --------------------------------------------------------
      
      if (
        model_name ==
        "Logistic Regression"
      ) {
        
        model_data <- data.frame(
          y = y_train,
          X_train,
          check.names = FALSE
        )
        
        model <- glm(
          y ~ .,
          data = model_data,
          family = binomial()
        )
        
        
        # --------------------------------------------------------
        # Random Forest
        # --------------------------------------------------------
        
      } else if (
        model_name ==
        "Random Forest"
      ) {
        
        model <- randomForest(
          x = X_train,
          y = y_factor,
          ntree = 500
        )
        
        
        # --------------------------------------------------------
        # SVM Radial
        # --------------------------------------------------------
        
      } else if (
        model_name ==
        "SVM Radial"
      ) {
        
        model <- kernlab::ksvm(
          x = X_train,
          y = y_factor,
          kernel = "rbfdot",
          prob.model = TRUE,
          scaled = TRUE
        )
        
        
        # --------------------------------------------------------
        # SVM Linear
        # --------------------------------------------------------
        
      } else if (
        model_name ==
        "SVM Linear"
      ) {
        
        model <- kernlab::ksvm(
          x = X_train,
          y = y_factor,
          kernel = "vanilladot",
          prob.model = TRUE,
          scaled = TRUE
        )
        
        
        # --------------------------------------------------------
        # Elastic Net
        # --------------------------------------------------------
        
      } else if (
        model_name ==
        "Elastic Net"
      ) {
        
        if (
          is.null(groups_train)
        ) {
          stop(
            "groups_train is required for Elastic Net."
          )
        }
        
        inner_fold_id <-
          make_group_stratified_folds(
            y = y_train,
            groups = groups_train,
            n_folds = 3,
            seed = current_seed + 777
          )
        
        model <- glmnet::cv.glmnet(
          x = as.matrix(X_train),
          y = y_train,
          family = "binomial",
          alpha = 0.5,
          foldid = inner_fold_id,
          type.measure = "deviance",
          standardize = TRUE
        )
        
        
        # --------------------------------------------------------
        # Decision Tree
        # --------------------------------------------------------
        
      } else if (
        model_name ==
        "Decision Tree"
      ) {
        
        model_data <- data.frame(
          y = y_factor,
          X_train,
          check.names = FALSE
        )
        
        model <- rpart::rpart(
          y ~ .,
          data = model_data,
          method = "class",
          control = rpart::rpart.control(
            cp = 0.001,
            minsplit = 10,
            xval = 0
          )
        )
        
        
        # --------------------------------------------------------
        # XGBoost
        # --------------------------------------------------------
        
      } else if (
        model_name ==
        "XGBoost"
      ) {
        
        dtrain <- xgboost::xgb.DMatrix(
          data = as.matrix(X_train),
          label = y_train
        )
        
        params <- list(
          objective = "binary:logistic",
          eval_metric = "logloss",
          max_depth = 6,
          eta = 0.1,
          subsample = 0.8,
          colsample_bytree = 0.8
        )
        
        model <- xgboost::xgb.train(
          params = params,
          data = dtrain,
          nrounds = 100,
          verbose = 0
        )
        
      } else {
        
        stop(
          paste(
            "Unknown model:",
            model_name
          )
        )
      }
      
      model
    }
    
    
    # ==========================================================
    # 24. MODEL PREDICTION
    # ==========================================================
    
    predict_model <- function(
    model_name,
    model,
    X_new
    ) {
      
      if (
        model_name ==
        "Logistic Regression"
      ) {
        
        new_data <- data.frame(
          X_new,
          check.names = FALSE
        )
        
        prob <- predict(
          model,
          newdata = new_data,
          type = "response"
        )
        
        
      } else if (
        model_name ==
        "Random Forest"
      ) {
        
        pred <- predict(
          model,
          newdata = X_new,
          type = "prob"
        )
        
        prob <- pred[
          ,
          "1"
        ]
        
        
      } else if (
        model_name %in%
        c(
          "SVM Radial",
          "SVM Linear"
        )
      ) {
        
        prob <- tryCatch(
          
          {
            
            p <- predict(
              model,
              newdata = X_new,
              type = "probabilities"
            )
            
            if (
              is.matrix(p) ||
              is.data.frame(p)
            ) {
              
              if (
                "1" %in%
                colnames(p)
              ) {
                
                p[, "1"]
                
              } else {
                
                p[, ncol(p)]
              }
              
            } else {
              
              as.numeric(p)
            }
          },
          
          error = function(e) {
            
            pred_class <- predict(
              model,
              newdata = X_new
            )
            
            as.numeric(
              pred_class == "1"
            )
          }
        )
        
        
      } else if (
        model_name ==
        "Elastic Net"
      ) {
        
        prob <- predict(
          model,
          newx = as.matrix(X_new),
          s = "lambda.min",
          type = "response"
        )
        
        prob <- as.numeric(
          prob
        )
        
        
      } else if (
        model_name ==
        "Decision Tree"
      ) {
        
        new_data <- data.frame(
          X_new,
          check.names = FALSE
        )
        
        pred <- predict(
          model,
          newdata = new_data,
          type = "prob"
        )
        
        if (
          is.matrix(pred) ||
          is.data.frame(pred)
        ) {
          
          if (
            "1" %in% colnames(pred)
          ) {
            prob <- pred[, "1"]
          } else {
            prob <- pred[, ncol(pred)]
          }
          
        } else {
          prob <- as.numeric(pred)
        }
        
        
      } else if (
        model_name ==
        "XGBoost"
      ) {
        
        dtest <- xgboost::xgb.DMatrix(
          data = as.matrix(X_new)
        )
        
        prob <- predict(
          model,
          newdata = dtest
        )
        
      } else {
        
        stop(
          paste(
            "Unknown model:",
            model_name
          )
        )
      }
      
      prob <- as.numeric(
        prob
      )
      
      prob[
        !is.finite(prob)
      ] <- NA_real_
      
      finite_idx <- is.finite(
        prob
      )
      
      prob[
        finite_idx
      ] <- pmin(
        pmax(
          prob[
            finite_idx
          ],
          0
        ),
        1
      )
      
      prob
    }
    
    
    # ==========================================================
    # 25. MODEL NAMES
    # ==========================================================
    
    model_names <- c(
      "Logistic Regression",
      "Elastic Net",
      "SVM Linear",
      "SVM Radial",
      "Random Forest",
      "Decision Tree",
      "XGBoost"
    )
    
    
    # ==========================================================
    # 26. OOF PREDICTIONS
    # ==========================================================
    
    oof_predictions <- matrix(
      NA_real_,
      nrow = length(y_training),
      ncol = length(model_names)
    )
    
    colnames(
      oof_predictions
    ) <- model_names
    
    
    # ==========================================================
    # 27. STORE FOLD RESULTS
    # ==========================================================
    
    fold_selected_features <-
      vector(
        "list",
        n_folds
      )
    
    fold_selected_indices <-
      vector(
        "list",
        n_folds
      )
    
    fold_uqi_history <-
      vector(
        "list",
        n_folds
      )
    
    cv_model_errors <-
      vector(
        "list",
        n_folds
      )
    
    
    # ==========================================================
    # 28. GROUP-WISE 5-FOLD CROSS-VALIDATION
    # ==========================================================
    
    for (
      fold in seq_len(
        n_folds
      )
    ) {
      
      fold_train_idx <- which(
        fold_id != fold
      )
      
      fold_valid_idx <- which(
        fold_id == fold
      )
      
      X_fold_train <- X_training[
        fold_train_idx,
        ,
        drop = FALSE
      ]
      
      y_fold_train <- y_training[
        fold_train_idx
      ]
      
      group_fold_train <-
        group_training[
          fold_train_idx
        ]
      
      X_fold_valid <- X_training[
        fold_valid_idx,
        ,
        drop = FALSE
      ]
      
      y_fold_valid <- y_training[
        fold_valid_idx
      ]
      
      
      # --------------------------------------------------------
      # GROUP LEAKAGE AUDIT
      # --------------------------------------------------------
      
      overlap <- intersect(
        unique(
          group_fold_train
        ),
        unique(
          group_training[
            fold_valid_idx
          ]
        )
      )
      
      if (
        length(overlap) > 0
      ) {
        stop(
          paste0(
            "GROUP LEAKAGE DETECTED in fold ",
            fold
          )
        )
      }
      
      
      # --------------------------------------------------------
      # Training-only imputation
      # --------------------------------------------------------
      
      fold_medians <-
        compute_column_medians(
          X_fold_train
        )
      
      X_fold_train <-
        apply_median_imputation(
          X_fold_train,
          fold_medians
        )
      
      X_fold_valid <-
        apply_median_imputation(
          X_fold_valid,
          fold_medians
        )
      
      
      # --------------------------------------------------------
      # UQI feature selection
      # --------------------------------------------------------
      
      uqi_result <-
        select_features_uqi(
          
          X_train =
            X_fold_train,
          
          y_train =
            y_fold_train,
          
          top_k =
            top_k,
          
          n_iter =
            n_iter,
          
          lr =
            lr,
          
          epsilon =
            epsilon,
          
          device =
            device,
          
          verbose =
            FALSE,
          
          uqi_seed =
            current_seed + fold
        )
      
      
      fold_selected_features[[fold]] <-
        uqi_result$selected_features
      
      fold_selected_indices[[fold]] <-
        uqi_result$selected_indices
      
      fold_uqi_history[[fold]] <-
        uqi_result
      
      
      selected_names <-
        uqi_result$selected_features
      
      selected_names <-
        intersect(
          selected_names,
          colnames(
            X_fold_valid
          )
        )
      
      if (
        length(selected_names) == 0
      ) {
        stop(
          paste(
            "No selected features available in fold",
            fold
          )
        )
      }
      
      
      X_fold_train_selected <-
        X_fold_train[
          ,
          selected_names,
          drop = FALSE
        ]
      
      X_fold_valid_selected <-
        X_fold_valid[
          ,
          selected_names,
          drop = FALSE
        ]
      
      
      # --------------------------------------------------------
      # Fit models
      # --------------------------------------------------------
      
      cv_model_errors[[fold]] <- list()
      
      for (
        m in seq_along(
          model_names
        )
      ) {
        
        model_name <-
          model_names[m]
        
        model <- tryCatch(
          
          {
            
            fit_model(
              model_name =
                model_name,
              X_train =
                X_fold_train_selected,
              y_train =
                y_fold_train,
              groups_train =
                group_fold_train
            )
            
          },
          
          error = function(e) {
            
            cv_model_errors[[fold]][[model_name]] <<-
              paste0(
                "FIT ERROR: ",
                conditionMessage(e)
              )
            
            NULL
          }
        )
        
        
        if (
          !is.null(model)
        ) {
          
          pred <- tryCatch(
            
            {
              
              predict_model(
                model_name =
                  model_name,
                model =
                  model,
                X_new =
                  X_fold_valid_selected
              )
              
            },
            
            error = function(e) {
              
              cv_model_errors[[fold]][[model_name]] <<-
                paste0(
                  "PREDICTION ERROR: ",
                  conditionMessage(e)
                )
              
              rep(
                NA_real_,
                length(
                  fold_valid_idx
                )
              )
            }
          )
          
          oof_predictions[
            fold_valid_idx,
            m
          ] <- pred
          
        } else {
          
          oof_predictions[
            fold_valid_idx,
            m
          ] <- NA_real_
        }
      }
      
      
      if (
        isTRUE(verbose)
      ) {
        
        cat(
          "Completed group-wise fold:",
          fold,
          "/",
          n_folds,
          "\n"
        )
      }
    }
    
    
    # ==========================================================
    # 29. CV RESULTS
    # ==========================================================
    
    cv_results <- vector(
      "list",
      length(model_names)
    )
    
    names(cv_results) <-
      model_names
    
    for (
      m in seq_along(
        model_names
      )
    ) {
      
      cv_results[[m]] <-
        calculate_metrics(
          y_training,
          oof_predictions[
            ,
            m
          ]
        )
    }
    
    cv_results_table <-
      do.call(
        rbind,
        cv_results
      )
    
    cv_results_table$Model <-
      rownames(
        cv_results_table
      )
    
    rownames(
      cv_results_table
    ) <- NULL
    
    cv_results_table <-
      cv_results_table[
        ,
        c(
          "Model",
          "Accuracy",
          "Balanced_Accuracy",
          "Sensitivity",
          "Specificity",
          "Precision_Class1",
          "Precision_Class0",
          "F1_Class1",
          "F1_Class0",
          "GMEAN",
          "MCC",
          "AUC"
        ),
        drop = FALSE
      ]
    
    
    # ==========================================================
    # 30. FINAL TRAINING / EXTERNAL IMPUTATION
    # ==========================================================
    
    training_medians <-
      compute_column_medians(
        X_training
      )
    
    X_training <-
      apply_median_imputation(
        X_training,
        training_medians
      )
    
    X_external <-
      apply_median_imputation(
        X_external,
        training_medians
      )
    
    
    # ==========================================================
    # 31. FINAL UQI SELECTION
    # ==========================================================
    
    final_uqi <-
      select_features_uqi(
        
        X_train =
          X_training,
        
        y_train =
          y_training,
        
        top_k =
          top_k,
        
        n_iter =
          n_iter,
        
        lr =
          lr,
        
        epsilon =
          epsilon,
        
        device =
          device,
        
        verbose =
          FALSE,
        
        uqi_seed =
          current_seed + 1000
      )
    
    
    final_selected_features <-
      final_uqi$selected_features
    
    final_selected_indices <-
      final_uqi$selected_indices
    
    final_weights <-
      final_uqi$weights
    
    
    # ==========================================================
    # 32. FINAL TRAINING DATA
    # ==========================================================
    
    final_selected_features <-
      intersect(
        final_selected_features,
        colnames(
          X_training
        )
      )
    
    X_final_train <-
      X_training[
        ,
        final_selected_features,
        drop = FALSE
      ]
    
    X_final_external <-
      X_external[
        ,
        final_selected_features,
        drop = FALSE
      ]
    
    
    # ==========================================================
    # 33. FINAL MODELS
    # ==========================================================
    
    final_models <- vector(
      "list",
      length(model_names)
    )
    
    names(final_models) <-
      model_names
    
    external_predictions <- matrix(
      NA_real_,
      nrow = length(
        y_external
      ),
      ncol = length(
        model_names
      )
    )
    
    colnames(
      external_predictions
    ) <- model_names
    
    external_model_errors <-
      list()
    
    
    # ==========================================================
    # 34. FINAL MODEL TRAINING
    # ==========================================================
    
    for (
      m in seq_along(
        model_names
      )
    ) {
      
      model_name <-
        model_names[m]
      
      model <- tryCatch(
        
        {
          
          fit_model(
            model_name =
              model_name,
            X_train =
              X_final_train,
            y_train =
              y_training,
            groups_train =
              group_training
          )
          
        },
        
        error = function(e) {
          
          external_model_errors[[model_name]] <<-
            paste0(
              "FIT ERROR: ",
              conditionMessage(e)
            )
          
          NULL
        }
      )
      
      final_models[[m]] <- list(model)
      
      if (
        !is.null(model)
      ) {
        
        external_predictions[
          ,
          m
        ] <-
          tryCatch(
            
            {
              
              predict_model(
                model_name =
                  model_name,
                model =
                  model,
                X_new =
                  X_final_external
              )
              
            },
            
            error = function(e) {
              
              external_model_errors[[model_name]] <<-
                paste0(
                  "PREDICTION ERROR: ",
                  conditionMessage(e)
                )
              
              rep(
                NA_real_,
                length(
                  y_external
                )
              )
            }
          )
        
      } else {
        
        external_predictions[
          ,
          m
        ] <-
          NA_real_
      }
    }
    
    
    # ==========================================================
    # 35. EXTERNAL RESULTS
    # ==========================================================
    
    external_results <- vector(
      "list",
      length(model_names)
    )
    
    names(external_results) <-
      model_names
    
    for (
      m in seq_along(
        model_names
      )
    ) {
      
      external_results[[m]] <-
        calculate_metrics(
          y_external,
          external_predictions[
            ,
            m
          ]
        )
    }
    
    external_results_table <-
      do.call(
        rbind,
        external_results
      )
    
    external_results_table$Model <-
      rownames(
        external_results_table
      )
    
    rownames(
      external_results_table
    ) <- NULL
    
    external_results_table <-
      external_results_table[
        ,
        c(
          "Model",
          "Accuracy",
          "Balanced_Accuracy",
          "Sensitivity",
          "Specificity",
          "Precision_Class1",
          "Precision_Class0",
          "F1_Class1",
          "F1_Class0",
          "GMEAN",
          "MCC",
          "AUC"
        ),
        drop = FALSE
      ]
    
    
    # ==========================================================
    # Store results of this repeat
    # ==========================================================
    
    all_cv_tables[[rep]] <- cv_results_table
    all_external_tables[[rep]] <- external_results_table
    
    last_result <- list(
      cv_results_table = cv_results_table,
      external_results_table = external_results_table,
      final_selected_features = final_selected_features,
      final_weights = final_weights,
      final_uqi = final_uqi,
      class_levels = original_levels,
      class_counts = class_counts,
      sample_ids = sample_ids,
      group_id = group_id,
      training_indices = training_indices,
      external_indices = external_indices,
      y_training = y_training,
      y_external = y_external,
      group_training = group_training,
      group_external = group_external,
      fold_id = fold_id,
      oof_predictions = oof_predictions,
      external_predictions = external_predictions,
      final_models = final_models,
      weight_table = data.frame(
        Feature = final_uqi$feature_names,
        Weight = final_weights,
        stringsAsFactors = FALSE
      ),
      cv_model_errors = cv_model_errors,
      external_model_errors = external_model_errors,
      imputation_medians = training_medians
    )
    
    last_result$weight_table <-
      last_result$weight_table[
        order(
          last_result$weight_table$Weight,
          decreasing = TRUE,
          na.last = TRUE
        ),
        ,
        drop = FALSE
      ]
    
  } # end of repeat loop
  
  
  # ==========================================================
  # AVERAGE RESULTS ACROSS REPEATS
  # ==========================================================
  
  metric_cols <- c(
    "Accuracy",
    "Balanced_Accuracy",
    "Sensitivity",
    "Specificity",
    "Precision_Class1",
    "Precision_Class0",
    "F1_Class1",
    "F1_Class0",
    "GMEAN",
    "MCC",
    "AUC"
  )
  
  # Average CV table
  avg_cv <- all_cv_tables[[1]]
  for (col in metric_cols) {
    vals <- sapply(all_cv_tables, function(tb) tb[[col]])
    avg_cv[[col]] <- rowMeans(vals, na.rm = TRUE)
  }
  
  # Average External table
  avg_external <- all_external_tables[[1]]
  for (col in metric_cols) {
    vals <- sapply(all_external_tables, function(tb) tb[[col]])
    avg_external[[col]] <- rowMeans(vals, na.rm = TRUE)
  }
  
  
  # ==========================================================
  # SAVE AVERAGED RESULTS
  # ==========================================================
  
  model_results_table <- rbind(
    
    cbind(
      Evaluation = "CV",
      avg_cv
    ),
    
    cbind(
      Evaluation = "External",
      avg_external
    )
  )
  
  write.csv(
    model_results_table,
    file = "model_results.csv",
    row.names = FALSE
  )
  
  
  # ==========================================================
  # PRINT SUMMARY
  # ==========================================================
  
  cat(
    "\n============================================================\n"
  )
  cat(
    "AVERAGED RESULTS OVER", n_repeats, "REPEATS\n"
  )
  cat(
    "============================================================\n"
  )
  
  cat("\n--- CV (averaged) ---\n")
  print(avg_cv, row.names = FALSE)
  
  cat("\n--- External (averaged) ---\n")
  print(avg_external, row.names = FALSE)
  
  
  # ==========================================================
  # RETURN
  # ==========================================================
  
  result <- list(
    
    n_repeats = n_repeats,
    
    # Averaged tables
    cv_results_table = avg_cv,
    external_results_table = avg_external,
    
    all_cv_tables = all_cv_tables,
    all_external_tables = all_external_tables,
    
    # From last repeat (for convenience)
    final_selected_features = last_result$final_selected_features,
    final_weights = last_result$final_weights,
    weight_table = last_result$weight_table,
    
    class_levels = last_result$class_levels,
    class_counts = last_result$class_counts,
    
    cv_model_errors = last_result$cv_model_errors,
    external_model_errors = last_result$external_model_errors,
    
    all_results = list(
      CV = avg_cv,
      External = avg_external
    )
  )
  
  return(result)
}