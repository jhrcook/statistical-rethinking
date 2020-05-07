
# Turn a PI into a tibble.
pi_to_df <- function(list_pi) {
    list_pi %>%
        t() %>%
        as.data.frame() %>%
        janitor::clean_names() %>%
        as_tibble()
}



scale_nums <- function(x, na.rm = FALSE) {
    (x - mean(x, na.rm = na.rm)) / sd(x, na.rm = na.rm)
}
