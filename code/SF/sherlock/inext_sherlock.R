#!/usr/bin/env Rscript
# ============================================================================
# iNEXT (incidence-frequency) for ONE hexagon scale x ONE guild, parallel across hexagons.
#
# Usage:
#   Rscript inext_sherlock.R <scale> <guild> <input_dir> <output_dir> [nboot] [level] [min_checklists] [date_tag]
#   e.g. Rscript inext_sherlock.R sf_250 trees $SCRATCH/inext/input $SCRATCH/inext/results/sf_250 100 0.8 5 11_21_26
#
#   guild: trees | water | shrubgrass | urban | native | nonnative | all
#
# Input (exported from P3_inext.ipynb):
#   <input_dir>/inc_<scale>_<guild>.csv     species x hex_id; first data row = n_sampling_units (checklists)
#   <input_dir>/inc_grade_<scale>_all.csv   species x grade  (only needed for guild = all)
#
# Output, every guild (same as the notebook's run_inext_group):
#   <scale>_ebird_<guild>_inc_covstd_<level>.csv   diversity at fixed coverage `level`, q = 0, 1, 2
#   <scale>_ebird_<guild>_inc_datainfo.csv         sample coverage etc. per hexagon
#   <scale>_ebird_<guild>_inc_failed.csv           hexagons iNEXT could not estimate (if any)
#
# Extra output for guild = all (same as the earlier all-species run):
#   SF_<size>hex_inc_{datainfo,asymp,sizebased,covbased,covstd}_{grade,hex}_<date>.csv
#   + grade rarefaction plot and .rds of the grade iNEXT object
# ============================================================================
suppressPackageStartupMessages({
  library(iNEXT); library(parallel)
  library(ggplot2); library(dplyr); library(stringr); library(scales)
})

args    <- commandArgs(trailingOnly = TRUE)
scale   <- args[1]
guild   <- args[2]
in_dir  <- args[3]
out_dir <- args[4]
nboot   <- if (length(args) >= 5) as.integer(args[5]) else 100L
level   <- if (length(args) >= 6) as.numeric(args[6]) else 0.8
min_chk <- if (length(args) >= 7) as.integer(args[7]) else 5L
date_tg <- if (length(args) >= 8) args[8] else format(Sys.Date(), "%m_%d_%y")
cores   <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", unset = detectCores()))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
RNGkind("L'Ecuyer-CMRG"); set.seed(123)   # reproducible random streams across forked workers

cat(sprintf("[%s] scale=%s guild=%s nboot=%d level=%.2f cores=%d\n",
            Sys.time(), scale, guild, nboot, level, cores))

read_inc <- function(path) {
  m <- read.csv(path, row.names = 1, check.names = FALSE)
  m[is.na(m)] <- 0
  m
}
x <- read_inc(file.path(in_dir, sprintf("inc_%s_%s.csv", scale, guild)))
cat(sprintf("Matrix: %d species x %d hexagons\n", nrow(x) - 1, ncol(x)))
if (ncol(x) == 0) { cat("No hexagons, nothing to do\n"); quit(save = "no") }

# ============================================================================
# 1. Every guild: diversity at a FIXED coverage level for each hexagon
#    (same as run_inext_group in the notebook, but hexagons run in parallel)
# ============================================================================
label <- sprintf("%s_ebird_%s_inc", scale, guild)

one_hex <- function(j) {
  hex <- colnames(x)[j]
  y   <- as.numeric(x[, j])
  y   <- y[c(TRUE, y[-1] > 0)]                 # keep row 1 + detected species only
  Tn  <- y[1]; inc <- y[-1]
  Q1  <- sum(inc == 1); Q2 <- sum(inc == 2)
  tryCatch(suppressWarnings({
    # coverage reachable at 2x the sample (to flag over-extrapolation)
    sc2T <- estimateD(list(h = y), datatype = "incidence_freq", base = "size",
                      level = 2 * Tn, q = 0, nboot = 0)$SC[1]
    d <- estimateD(list(h = y), datatype = "incidence_freq", base = "coverage",
                   level = level, q = c(0, 1, 2), nboot = nboot)
    d$Assemblage <- hex
    d$T_checklists <- Tn; d$S_obs <- length(inc); d$Q1 <- Q1; d$Q2 <- Q2
    d$sc2T <- sc2T; d$beyond_2T <- sc2T < level
    list(d = d, fail = NULL)
  }), error = function(e) {
    list(d = NULL, fail = data.frame(hex_id = hex, T_checklists = Tn, S_obs = length(inc),
                                     Q1 = Q1, Q2 = Q2, error = conditionMessage(e)))
  })
}

t0 <- Sys.time()
res     <- mclapply(seq_len(ncol(x)), one_hex, mc.cores = cores, mc.preschedule = FALSE)
cov_std <- bind_rows(lapply(res, `[[`, "d"))
fails   <- bind_rows(lapply(res, `[[`, "fail"))

write.csv(cov_std, file.path(out_dir, paste0(label, "_covstd_", level, ".csv")), row.names = FALSE)
if (nrow(fails)) write.csv(fails, file.path(out_dir, paste0(label, "_failed.csv")), row.names = FALSE)
write.csv(suppressWarnings(DataInfo(x, datatype = "incidence_freq")),
          file.path(out_dir, paste0(label, "_datainfo.csv")), row.names = FALSE)

n_ok <- if (nrow(cov_std)) length(unique(cov_std$Assemblage)) else 0
n_2T <- if (nrow(cov_std)) sum(cov_std$beyond_2T[cov_std$Order.q == 0]) else 0
cat(sprintf("%-32s %4d ok | %3d failed | %3d beyond 2T at SC=%.2f | %.1f min\n",
            label, n_ok, nrow(fails), n_2T, level, as.numeric(difftime(Sys.time(), t0, units = "mins"))))

if (guild != "all") { cat(sprintf("[%s] done\n", Sys.time())); quit(save = "no") }

# ============================================================================
# 2. All species only: full curves, common-coverage estimates, grade plot
# ============================================================================
prefix <- paste0("SF_", sub("sf_", "", scale), "hex")
f_out  <- function(what, lab) file.path(out_dir, paste0(prefix, "_inc_", what, "_", lab, "_", date_tg, ".csv"))
write_set <- function(out, cs, lab) {
  write.csv(out$DataInfo,                f_out("datainfo",  lab), row.names = FALSE)
  write.csv(out$AsyEst,                  f_out("asymp",     lab), row.names = FALSE)
  write.csv(out$iNextEst$size_based,     f_out("sizebased", lab), row.names = FALSE)
  write.csv(out$iNextEst$coverage_based, f_out("covbased",  lab), row.names = FALSE)
  write.csv(cs,                          f_out("covstd",    lab), row.names = FALSE)
}

# 2a. Grade level (4 assemblages)
inc_grade <- read_inc(file.path(in_dir, sprintf("inc_grade_%s_all.csv", scale)))
t0 <- Sys.time()
inext_grade <- iNEXT(inc_grade, datatype = "incidence_freq", q = c(0, 1, 2), nboot = nboot,
                     knots = 200)   # more points per curve -> smooth plots (only 4 curves, so cheap)
cov_grade   <- estimateD(inc_grade, datatype = "incidence_freq", base = "coverage",
                         q = c(0, 1, 2), level = NULL, nboot = nboot)
write_set(inext_grade, cov_grade, "grade")
saveRDS(inext_grade, file.path(out_dir, paste0(prefix, "_inext_grade_", date_tg, ".rds")))
cat(sprintf("Grade curves done in %.1f min; common coverage = %s\n",
            as.numeric(difftime(Sys.time(), t0, units = "mins")),
            paste(round(unique(cov_grade$SC), 3), collapse = ", ")))

# 2b. Hexagon curves (hexagons with >= min_chk checklists and >= 2 species)
ok   <- (unlist(x[1, ]) >= min_chk) & (colSums(x[-1, , drop = FALSE] > 0) >= 2)
xh   <- x[, ok, drop = FALSE]
cat(sprintf("Hexagon curves: %d of %d hexagons (>= %d checklists)\n", sum(ok), length(ok), min_chk))
site <- function(j) xh[, j, drop = FALSE]

#     common coverage, same rule as estimateD(level = NULL):
#     lowest coverage any hexagon reaches at 2x its own number of checklists
cov_2T <- unlist(mclapply(seq_len(ncol(xh)), function(j) {
  s <- site(j)
  estimateD(s, datatype = "incidence_freq", base = "size", level = 2 * s[1, 1], q = 0, nboot = 0)$SC[1]
}, mc.cores = cores))
cov_level <- min(cov_2T, na.rm = TRUE)
cat(sprintf("Common coverage level for hexagons: %.3f\n", cov_level))

t0 <- Sys.time()
res2 <- mclapply(seq_len(ncol(xh)), function(j) {
  s <- site(j)
  tryCatch(list(out = iNEXT(s, datatype = "incidence_freq", q = c(0, 1, 2), nboot = nboot),
                cs  = estimateD(s, datatype = "incidence_freq", base = "coverage",
                                q = c(0, 1, 2), level = cov_level, nboot = nboot)),
           error = function(e) { message("hex ", colnames(s), ": ", conditionMessage(e)); NULL })
}, mc.cores = cores, mc.preschedule = FALSE)
failed2 <- colnames(xh)[vapply(res2, is.null, logical(1))]
if (length(failed2)) cat("Failed hexagon curves:", paste(failed2, collapse = ", "), "\n")
res2 <- Filter(Negate(is.null), res2)

combined <- list(
  DataInfo = bind_rows(lapply(res2, function(r) r$out$DataInfo)),
  AsyEst   = bind_rows(lapply(res2, function(r) r$out$AsyEst)),
  iNextEst = list(size_based     = bind_rows(lapply(res2, function(r) r$out$iNextEst$size_based)),
                  coverage_based = bind_rows(lapply(res2, function(r) r$out$iNextEst$coverage_based))))
write_set(combined, bind_rows(lapply(res2, function(r) r$cs)), "hex")
cat(sprintf("Hexagon curves done in %.1f min (%d ok, %d failed)\n",
            as.numeric(difftime(Sys.time(), t0, units = "mins")), length(res2), length(failed2)))

# 2c. Grade plots (HOLC colours), one PNG per iNEXT curve type
#     type 1: diversity vs number of checklists     (sample-size-based R/E)
#     type 2: sample coverage vs number of checklists (sample completeness)
#     type 3: diversity vs sample coverage            (coverage-based R/E)
holc_colors <- c("A" = "#9FE2BF", "B" = "#6495ED", "C" = "#FFBF00", "D" = "#DE3163")
holc_shapes <- c("A" = 16, "B" = 17, "C" = 15, "D" = 4)

plot_grade_curves <- function(inext_obj, type, title = "") {
  p_data <- fortify(inext_obj, type = type)
  p_data$grade <- str_sub(p_data$Assemblage, -1, -1)
  # observed sample = end of rarefaction, start of extrapolation
  obs <- p_data %>% filter(Method == "Observed") %>% group_by(Assemblage, Order.q) %>% slice(1) %>% ungroup()

  p <- ggplot(p_data, aes(x = x, y = y, colour = grade, fill = grade)) +
    geom_ribbon(aes(ymin = y.lwr, ymax = y.upr), alpha = 0.2, colour = NA) +
    geom_line(aes(linetype = Method), linewidth = 1.5) +
    geom_point(data = obs, aes(shape = grade), size = 4) +
    scale_colour_manual(values = holc_colors, name = "Grade") +
    scale_fill_manual(values = holc_colors, name = "Grade") +
    scale_shape_manual(values = holc_shapes, name = "Grade") +
    scale_linetype_manual(values = c("Rarefaction" = "solid", "Observed" = "solid",
                                     "Extrapolation" = "dashed"),
                          breaks = c("Rarefaction", "Extrapolation"), name = "Method") +
    theme_light(base_size = 22) + theme(panel.grid.major = element_blank()) +
    labs(title = title)

  if (type != 2)   # coverage (type 2) is the same for every q, so only one panel there
    p <- p + facet_wrap(~ Order.q, scales = "fixed",
                        labeller = labeller(Order.q = c("0" = "Richness", "1" = "Shannon's", "2" = "Simpson's")))
  if (type == 1) p <- p + labs(x = "Number of checklists (1000)", y = "Species diversity") +
                   scale_x_continuous(labels = unit_format(unit = " ", scale = 1e-3))
  if (type == 2) p <- p + labs(x = "Number of checklists (log scale)", y = "Sample coverage") +
                   scale_x_log10(labels = comma) +          # coverage rises within the first checklists
                   scale_y_continuous(labels = percent, limits = c(NA, 1))
  if (type == 3) p <- p + labs(x = "Sample coverage", y = "Species diversity") +
                   scale_x_continuous(labels = percent)
  p
}

plot_names <- c("1" = "speciesaccum", "2" = "completeness", "3" = "coverage")
for (t in 1:3) {
  ggsave(file.path(out_dir, paste0(prefix, "_inext_", plot_names[[as.character(t)]], "_grades_", date_tg, ".png")),
         plot_grade_curves(inext_grade, t, title = scale),
         units = "in", width = if (t == 2) 9 else 13, height = 7, dpi = 300)
}

cat(sprintf("[%s] %s / %s finished\n", Sys.time(), scale, guild))
