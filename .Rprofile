
source_scripts <- function() {
    for (f in list.files("scripts", full.names = TRUE, pattern = "R$")) {
        source(f)
    }
}

home_r_profile <- file.path("~", ".Rprofile")
if (file.exists(home_r_profile)) {
    source(home_r_profile)
}
