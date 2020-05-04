
# Turn a PI into a tibble.
pi_to_df <- function(list_pi) {
    list_pi %>%
        t() %>%
        as.data.frame() %>%
        janitor::clean_names() %>%
        as_tibble()
}
