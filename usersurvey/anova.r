# ART_ANOVA_script.R
# Aligned Rank Transform (ART) two-way repeated-measures ANOVA

library(ARTool)
library(tidyverse)
library(emmeans)

# === 1) Load CSV ===
csv_file <- "Full text export Robot Planning Survey Release(2).xlsx - All Data.csv"
df <- read_csv(csv_file, show_col_types = FALSE)

# === 2) Identify AB Test columns and pair columns ===
cols <- colnames(df)
ab_pairs <- list()
for (i in seq_along(cols)) {
  if (str_detect(cols[i], "AB Test")) {
    if (i + 1 <= length(cols)) {
      ab_pairs[[length(ab_pairs) + 1]] <- c(ab_col = cols[i], q_col = cols[i + 1])
    }
  }
}

# === 3) Define the correct answers for each test ===
true_answers <- list(
  "1" = "Failure",
  "2" = "Failure",
  "3" = "Success",
  "4" = "Failure",
  "5" = "Failure"
)

# === 4) Convert responses into long format ===
rows <- list()
for (test_idx in seq_along(ab_pairs)) {
  pair <- ab_pairs[[test_idx]]
  ab_col <- pair["ab_col"]
  q_col  <- pair["q_col"]
  correct_answer <- true_answers[[as.character(test_idx)]]

  # Get non-NA rows for this test pair
  valid_rows <- which(!is.na(df[[ab_col]]) & !is.na(df[[q_col]]))

  if (length(valid_rows) > 0) {
    for (i in valid_rows) {
      ab <- df[[ab_col]][i]
      ans <- df[[q_col]][i]

      # Skip if either value is NA (shouldn't happen due to filtering, but safe)
      if (is.na(ab) || is.na(ans)) next

      correct <- as.integer(ans == correct_answer)
      category <- if (test_idx %in% c(1, 3)) "Logic" else "Consistency"

      rows[[length(rows) + 1]] <- tibble(
        Participant = i,  # Use original row number as participant ID
        Condition   = ab,
        Category    = category,
        Test        = test_idx,
        Correct     = correct
      )
    }
  }
}

# Check if we have any data before proceeding
if (length(rows) == 0) {
  stop("No valid data found. Check your column names and data structure.")
}

long_df <- bind_rows(rows)

# Debug: Check what values are in Condition column
print("Unique values in Condition column before recoding:")
print(unique(long_df$Condition))

# === 5) Rename conditions safely ===
# First, ensure Condition is a character vector, not NULL
if (!is.null(long_df$Condition)) {
  long_df$Condition <- dplyr::recode(long_df$Condition,
                                     "A:" = "Baseline",
                                     "B:" = "PerceptTwin")
} else {
  stop("Condition column is NULL. Check data processing in step 4.")
}

# Debug: Check recoding worked
print("Unique values in Condition column after recoding:")
print(unique(long_df$Condition))

# === 6) Aggregate to percent-correct per Participant × Condition × Category ===
agg <- long_df %>%
  group_by(Participant, Condition, Category) %>%
  summarize(Correct = mean(Correct) * 100, .groups = "drop")

# Make sure variables are factors
agg <- agg %>%
  mutate(
    Participant = factor(Participant),
    Condition   = factor(Condition, levels = c("Baseline", "PerceptTwin")),
    Category    = factor(Category)
  )

# Check data structure
print("Data structure:")
print(str(agg))
print(head(agg))

# Check for missing data patterns
print("Number of observations per participant:")
part_counts <- table(agg$Participant)
print(part_counts)

# === 7) NEW: Handle missing data - keep all participants regardless of completeness ===
# ART can handle missing data in repeated measures designs
print(paste("Total participants:", n_distinct(agg$Participant)))

# Check which condition-category combinations each participant has
participant_coverage <- agg %>%
  group_by(Participant) %>%
  summarize(
    n_conditions = n_distinct(Condition),
    n_categories = n_distinct(Category),
    n_combinations = n(),
    .groups = "drop"
  )

print("Participant coverage:")
print(participant_coverage)

# === 8) Run ART ANOVA on all available data ===
# ART can handle unbalanced designs with missing data
art_model <- art(Correct ~ Condition * Category + Error(Participant/(Condition*Category)),
                 data = agg)

art_anova <- anova(art_model)
print("=== ART ANOVA (aligned-and-ranked) ===")
print(art_anova)

# === 9) Follow-up pairwise contrasts ===
# Condition effect within each category
cond_lm <- artlm(art_model, "Condition")
emmeans_cond_by_cat <- emmeans(cond_lm, ~ Condition | Category)
pairs_cond_by_cat <- contrast(emmeans_cond_by_cat, method = "pairwise", adjust = "bonferroni")

print("=== Pairwise: Baseline vs PerceptTwin within each Category ===")
print(pairs_cond_by_cat)

# === 10) Plotting with participant count information ===
# Calculate how many participants contributed to each condition-category combination
n_participants <- agg %>%
  group_by(Condition, Category) %>%
  summarize(n = n_distinct(Participant), .groups = "drop")

ggplot(agg, aes(x = Category, y = Correct, fill = Condition)) +
  stat_summary(fun = mean, geom = "col", position = position_dodge(width = 0.7)) +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2,
               position = position_dodge(width = 0.7)) +
  geom_text(data = n_participants,
            aes(label = paste("n =", n), y = 5),
            position = position_dodge(width = 0.7),
            vjust = 0, size = 3) +
  theme_minimal(base_size = 14) +
  labs(title = "Mean percent correct by Condition and Category",
       subtitle = paste("Total N =", n_distinct(agg$Participant), "participants"),
       y = "Percent correct", x = "Category") +
  ylim(0, 100)