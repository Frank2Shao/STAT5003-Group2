# Member D: run from the repository root with Rscript R/04_eda_numerical.R
# Uses A/B's processed data unchanged; no imputation or model fitting during EDA.
required <- c('readr', 'dplyr', 'tidyr', 'ggplot2')
missing <- required[!vapply(required, requireNamespace, logical(1), quietly=TRUE)]
if (length(missing)) stop('Install packages: ', paste(missing, collapse=', '))
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
if (!file.exists('data/processed/acecqa_analysis.csv')) source('R/01_prepare_data.R')
d_data <- read_csv('data/processed/acecqa_analysis.csv', show_col_types=FALSE)
d_tiers <- c('Below NQS','Meeting NQS','Above NQS')
d_vars <- c('number_approved_places','service_age_years','provider_age_years','provider_service_count')
d_labels <- c('Approved places [log10(1+x)]','Service approval age [years]',
              'Provider approval age [years]','Provider service count [log10(1+x)]')
stopifnot(!anyDuplicated(d_data$service_id), !anyNA(d_data$rating_tier),
          all(d_data$rating_tier %in% d_tiers), all(d_vars %in% names(d_data)))
d_data$rating_tier <- factor(d_data$rating_tier, levels=d_tiers)
d_long <- d_data |> select(service_id,provider_id,service_type,rating_tier,all_of(d_vars)) |>
  pivot_longer(all_of(d_vars), names_to='variable',values_to='value')
d_summary <- function(x) {
  v <- x[is.finite(x)]
  if (!length(v)) return(tibble(n=length(x),observed=0L,missing=sum(is.na(x)),
    missing_percent=100*mean(is.na(x)),min=NA_real_,q1=NA_real_,median=NA_real_,
    q3=NA_real_,max=NA_real_,mean=NA_real_,sd=NA_real_,iqr_flags=0L,negative=0L,zero=0L))
  q <- quantile(v,c(.25,.5,.75),names=FALSE)
  tibble(n=length(x),observed=length(v),missing=sum(is.na(x)),
    missing_percent=100*mean(is.na(x)),min=min(v),q1=q[1],median=q[2],q3=q[3],
    max=max(v),mean=mean(v),sd=sd(v),
    iqr_flags=sum(v<q[1]-1.5*(q[3]-q[1]) | v>q[3]+1.5*(q[3]-q[1])),
    negative=sum(v<0),zero=sum(v==0))
}
d_profile <- d_long |> group_by(variable) |> group_modify(~d_summary(.x$value)) |> ungroup()
d_by_tier <- d_long |> group_by(variable,rating_tier) |> group_modify(~d_summary(.x$value)) |> ungroup()
d_by_type <- d_long |> group_by(variable,service_type,rating_tier) |>
  group_modify(~d_summary(.x$value)) |> ungroup()
d_missing_type <- d_long |> group_by(variable,service_type) |> summarise(
  n=n(),missing=sum(is.na(value)),missing_percent=100*mean(is.na(value)),.groups='drop')
# Service-level correlations: provider attributes are repeated for each linked service.
d_pairs <- combn(d_vars,2,simplify=FALSE)
d_cor <- bind_rows(lapply(d_pairs,function(v) {
  ok <- complete.cases(d_data[,v]); x <- d_data[[v[1]]][ok]; y <- d_data[[v[2]]][ok]
  tibble(var1=v[1],var2=v[2],n_complete=sum(ok),spearman=cor(x,y,method='spearman'))
}))
# Provider-level sensitivity summary; one record per represented provider.
d_providers <- d_data |> distinct(provider_id,provider_age_years,provider_service_count)
stopifnot(!anyDuplicated(d_providers$provider_id))
d_provider_profile <- d_providers |> pivot_longer(-provider_id,names_to='variable',values_to='value') |>
  group_by(variable) |> group_modify(~d_summary(.x$value)) |> ungroup()
# Fixed descriptive bins, not fitted cut points or mandatory modelling categories.
d_scale_mix <- d_data |> mutate(scale_band=cut(provider_service_count,
  breaks=c(0,1,5,20,100,Inf),labels=c('1','2-5','6-20','21-100','101+'))) |>
  count(scale_band,rating_tier,.drop=FALSE) |> group_by(scale_band) |>
  mutate(total=sum(n),percent=100*n/total) |> ungroup()
d_extremes <- d_long |> filter(!is.na(value)) |> group_by(variable) |>
  slice_max(value,n=5,with_ties=FALSE) |> ungroup()
d_plot_data <- d_long |> mutate(panel=factor(variable,levels=d_vars,labels=d_labels),
  plotted=if_else(variable %in% d_vars[c(1,4)],log10(1+pmax(value,0)),value))
d_cols <- c('Below NQS'='#C44E52','Meeting NQS'='#4C78A8','Above NQS'='#59A14F')
d_theme <- theme_minimal(base_size=11)+theme(legend.position='top',panel.grid.minor=element_blank(),
  strip.text=element_text(face='bold'),plot.title=element_text(face='bold'))
d_distribution_plot <- ggplot(filter(d_plot_data,!is.na(plotted)),aes(plotted))+
  geom_histogram(bins=35,fill='#4C78A8',colour='white',linewidth=.2)+
  facet_wrap(~panel,scales='free',ncol=2)+labs(title='Distributions of numerical predictors',
  subtitle='Labelled services; missing values excluded separately for each variable',x=NULL,y='Services')+d_theme
# Full range retained; transformed axes compress the tails without dropping services.
d_rating_plot <- ggplot(filter(d_plot_data,!is.na(plotted)),aes(rating_tier,plotted,fill=rating_tier))+
  geom_boxplot(width=.58,outlier.size=.45,outlier.alpha=.15)+
  facet_wrap(~panel,scales='free_y',ncol=4)+scale_fill_manual(values=d_cols)+
  labs(title='Numerical predictors by recorded NQS tier',
  subtitle='Full range retained; count variables shown as log10(1+x)',x=NULL,y=NULL)+d_theme+
  theme(legend.position='none', axis.text.y = element_text(size=7), axis.text.x = element_text(size=7))
d_type_plot <- ggplot(filter(d_plot_data,!is.na(plotted)),aes(rating_tier,plotted,fill=rating_tier))+
  geom_boxplot(width=.55,outlier.size=.3,outlier.alpha=.1)+
  facet_wrap(vars(panel,service_type),scales='free_y',ncol=2)+scale_fill_manual(values=d_cols)+
  labs(title='Sensitivity check within service type',x=NULL,y=NULL)+d_theme+theme(legend.position='none')
d_cor_grid <- bind_rows(d_cor |> select(var1,var2,spearman),
  d_cor |> select(var1=var2,var2=var1,spearman),tibble(var1=d_vars,var2=d_vars,spearman=1))
stopifnot(nrow(distinct(d_cor_grid,var1,var2)) == 16L)
d_short <- c('Places','Service age','Provider age','Provider scale')
d_cor_grid <- d_cor_grid |> mutate(var1=factor(var1,levels=d_vars,labels=d_short),
  var2=factor(var2,levels=rev(d_vars),labels=rev(d_short)))
d_correlation_plot <- ggplot(d_cor_grid,aes(var1,var2,fill=spearman))+geom_tile(colour='white')+
  geom_text(aes(label=sprintf('%.2f',spearman)),size=4)+
  scale_fill_gradient2(low='#C44E52',mid='white',high='#4C78A8',limits=c(-1,1))+
  coord_equal()+labs(title='Numerical predictor relationships',
  subtitle='Spearman correlations; pairwise complete services',x=NULL,y=NULL,fill='rho')+d_theme

dir.create('outputs/numerical',recursive=TRUE,showWarnings=FALSE)
dir.create('figures',showWarnings=FALSE)
for (nm in c('profile','by_tier','by_type','missing_type','cor','provider_profile','scale_mix','extremes'))
  write_csv(get(paste0('d_',nm)),paste0('outputs/numerical/',nm,'.csv'),na='')
ggsave('figures/numerical_distributions.png',d_distribution_plot,width=9,height=6,dpi=180)
ggsave('figures/numerical_by_rating.png',d_rating_plot,width=9,height=6,dpi=180)
ggsave('figures/numerical_by_service_type.png',d_type_plot,width=10,height=11,dpi=180)
ggsave('figures/numerical_correlations.png',d_correlation_plot,width=7,height=5,dpi=180)
print(d_profile,width=Inf)
print(d_by_tier |> select(variable,rating_tier,observed,missing,median,q1,q3),n=20)
print(d_cor)
print(d_missing_type)
print(d_provider_profile,width=Inf)
# Audit upstream definitions against raw registers (read all columns as text).
d_raw_s <- read_csv('Education-services-au-export.csv',col_types=cols(.default=col_character()))
d_raw_p <- read_csv('Approved-providers-au-export.csv',col_types=cols(.default=col_character()))
d_raw_scale <- d_raw_s |> count(`Provider Approval Number`,name='expected_scale')
d_check <- d_data |> left_join(d_raw_scale,by=c('provider_id'='Provider Approval Number'))
stopifnot(all(d_check$provider_service_count==d_check$expected_scale),
  nrow(d_data)==sum(!is.na(d_raw_s$OverallRating)),
  all(vapply(d_data[d_vars],function(x) all(is.finite(x)|is.na(x)),logical(1))))
d_date_peaks <- bind_rows(
  d_raw_s |> count(ServiceApprovalGrantedDate,name='n') |> rename(date=ServiceApprovalGrantedDate) |>
    mutate(register='All services'),
  d_raw_p |> count(`Date Approval Granted`,name='n') |> rename(date=`Date Approval Granted`) |>
    mutate(register='All providers')) |> group_by(register) |> slice_max(n,n=5,with_ties=FALSE) |> ungroup()
write_csv(d_date_peaks,'outputs/numerical/approval_date_peaks.csv')
