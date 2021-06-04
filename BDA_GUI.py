# Isakov, L., Lo, A. W., and Montazerhodjat, V. (2019) 
# Is the FDA too Conservative or too Aggressive?: 
# A Bayesian Decision Analysis of Clinical Trial Design,
# Journal of Econometrics

# Copyright (C) 2020 Zied Ben Chaouch's replication of
# Leah Isakov, Andrew W. Lo and Vahid Montazerhodjat, Journal of Econometrics 2019

# This program is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the 
# Free Software Foundation, either version 3 of the License, or (at your option) 
# any later version.
# This program is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# For detailed information on GNU General Public License, 
# please visit https://www.gnu.org/licenses/licenses.en.html

import pandas as pd
from scipy import optimize
from scipy.optimize import NonlinearConstraint
from scipy.stats import norm

import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider, ColumnDataSource, DataTable, TableColumn, Select, BasicTicker, \
    ColorBar, LinearColorMapper, CategoricalColorMapper, PrintfTickFormatter, Button, Legend
from bokeh.models.widgets import Div
from bokeh.plotting import figure
from bokeh.palettes import RdBu3

# Import Data:
My_Data_df = pd.read_csv('BDA_data_10.csv').set_index('Disease Name')

# Default Values
maxpower_0 = 0.9
p0_0 = 0.5
sigma_0 = 2.0
k_0 = 3
kappa0 = 100.0
c_f0 = 20.0

# Set up widgets
Disease_Name_gui = Select(title="Disease Considered", value='Ischemic heart disease', options=My_Data_df.index.tolist())
Disease_Rank_gui = My_Data_df.loc[Disease_Name_gui.value, 'YLL Rank']

max_power_check_gui = Select(title="Add Power Constraint: (No = 100% Max Power)", value="Yes", options=["Yes", "No"])
maxPower_gui = Slider(title="Maximal Power Allowed", value=maxpower_0, start=0.0, end=1.0, step=0.01, format="0.0%")
if max_power_check_gui.value == "No":
    maxPower_gui.value = 1.0

p_H0_gui = Slider(title="Probability of an Ineffective Treatment (p\u2080)", value=p0_0, start=0.0, end=1.0, step=0.01,
                  format="0.0%")
sigma_gui = Slider(title="Standard Deviation (\u03C3)", value=sigma_0, start=0.01, end=20.0, step=0.01,
                   format="0[.]000")
N_gui = Slider(title="Prevalence (N, in Thousands)", value=8895.61, start=0.1, end=10000.0, step=0.01, format="0[.]00")
c1_gui = Slider(title="Type-I Cost (c\u2081)", value=0.07, start=0.010, end=5.0, step=0.001, format="0[.]000")
s2_gui = Slider(title="Severity of Disease (s\u2082)", value=0.120, start=0.010, end=1.00, step=0.0005,
                format="0[.]000")
k_gui = Slider(title="Magnitude of Treatment Effect Parameter (n)", value=k_0, start=0, end=5, step=1)

trial_cost_check_gui = Select(title="Add Trial Costs: (No: \u03BA = 0.001, c_f = 0)", value="Yes",
                              options=["Yes", "No"])
kappa_gui = Slider(title="\u03BA (in Thousands)", value=kappa0, start=0.0, end=1000.0, step=0.5,
                   format="0[.]000")
cf_gui = Slider(title="Trial Cost Per Patient (c_f, in Thousands)", value=c_f0, start=0.0, end=500.0, step=0.5,
                format="0[.]0")
if trial_cost_check_gui.value == "No":
    kappa_gui.value = 0.001
    cf_gui.value = 0.0

submit = Button(label='Calculate', button_type='success', visible=True)

# BDA Calculations:

def Run_BDA(c_1, s_2, k, N, Disease_Name, Disease_Rank, sigma, p_H0, maxPower, kappa, c_f):
    # Data Cleaning:
    N = 1000 * N  # N is given in thousand
    kappa = 1000 * kappa  # kappa is given in thousand
    c_f = 1000 * c_f  # c_f is given in thousand

    # Helpful Definitions:

    # Magnitude of the Treatment Effect for an Effective Drug
    def delta_0_k(k, sigma):  # k = 0, 1, 2, 3
        return sigma * 2 ** (-k)

    # Incremental Cost Incurred by Adding an Extra Patient to Each Arm:
    def gamma_k(sigma, delta_0):
        return (4 * 10 ** (-3)) * (delta_0 / sigma)

    # Information per Trial
    def I_n(n, sigma):
        return n / (2 * sigma ** 2)

    # Expected Cost Function:
    def C_n(targets, k=k, N=N, p_H0=p_H0, c_1=c_1, s_2=s_2, sigma=sigma, kappa=kappa, c_f=c_f):
        # Parameters
        n, lambda_n = targets[0], targets[1]
        delta_0 = delta_0_k(k, sigma)
        c_2 = s_2 * min(1, delta_0 / sigma)
        p_H1 = 1.0 - p_H0  # Probability that the treatment is effective
        c_2_bar = (p_H1 / p_H0) * (c_2 / c_1)  # Normalized Type-II Cost

        # Helpers:
        term_A = N * norm.cdf(-lambda_n)
        term_B = N * c_2_bar * norm.cdf(lambda_n - delta_0 * (I_n(n, sigma)) ** 0.5)
        term_C = n * (1 + gamma_k(sigma, delta_0) * N * c_2_bar + c_f / (kappa * p_H0 * c_1))
        return kappa * p_H0 * c_1 * (term_A + term_B + term_C)

    # Find alpha Given the Critical Value lambda_n:
    def get_alpha(lambda_n):
        return 1 - norm.cdf(lambda_n)

    # Find Power Given the Critical Value lambda_n:
    def get_Power(lambda_n, n, k=k, sigma=sigma):
        delta_0 = delta_0_k(k, sigma)
        I = I_n(n, sigma)
        return norm.cdf(delta_0 * I ** 0.5 - lambda_n)

    # Define the Power Constraint: Power <= maxPower
    def power_constr(targets, maxPower=maxPower):
        return -(maxPower - get_Power(targets[1], targets[0]))

    # Find beta Given the Power: beta = 1 - beta
    def get_beta(Power):
        return 1 - Power

    # Calculate Optimal Sample Size & Critical Value:

    # Unconstrained Minimization:
    result_unc = optimize.minimize(C_n, np.array([700, 2]), method="Nelder-Mead")
    n_unc, lambda_unc = result_unc.x
    cost_unc = result_unc.fun
    alpha_unc, power_unc = get_alpha(lambda_unc), get_Power(lambda_unc, n_unc)
    EC1_unc = kappa * (N * c_1 * norm.cdf(-lambda_unc) + n_unc * c_1)  # Expected Type-I Cost
    c_2 = s_2 * min(1, delta_0_k(k, sigma) / sigma)
    EC2_unc = kappa * (N * c_2 * norm.cdf(lambda_unc - delta_0_k(k, sigma) * (I_n(n_unc, sigma)) ** 0.5) + \
                       n_unc * (gamma_k(sigma, delta_0_k(k, sigma)) * N * c_2 + c_f / kappa))  # Expected Type-II Cost

    # Constrained Maximization: with Power <= maxPower
#         if power_unc <= maxPower:
    if True:
        n_con, lambda_con = n_unc, lambda_unc
        cost_con = cost_unc
        alpha_con, power_con = alpha_unc, power_unc
        EC1_con = EC1_unc
        EC2_con = EC2_unc
    else:
        nlc = NonlinearConstraint(power_constr, -np.inf, 0)
        result_con = optimize.minimize(C_n, np.array([700, 2]), method='trust-constr', constraints=nlc)
        n_con, lambda_con = result_con.x
        cost_con = result_con.fun
        alpha_con, power_con = get_alpha(lambda_con), get_Power(lambda_con, n_con)

        EC1_con = kappa * (N * c_1 * norm.cdf(-lambda_unc) + n_unc * c_1)  # Expected Type-I Cost
        c_2 = s_2 * min(1, delta_0_k(k, sigma) / sigma)
        EC2_con = kappa * (N * c_2 * norm.cdf(lambda_unc - delta_0_k(k, sigma) * (I_n(n_unc, sigma)) ** 0.5) + \
                           n_unc * (gamma_k(sigma,
                                            delta_0_k(k, sigma)) * N * c_2 + c_f / kappa))  # Expected Type-II Cost

    # Collect the Results Into a Data Frame:
    results_df = pd.DataFrame(data=
                              {'YLL Rank': [Disease_Rank],
                               'Disease Name': [Disease_Name],
                               'Prevalence (Thousands)': [round(N / 1000, 2)],
                               'Severity': [round(s_2, 3)],
                               'Sample Size': [int(round(n_con, 0))],
                               'Critical Value': [round(lambda_con, 3)],
                               '\u03B1 (%)': [round(100 * alpha_con, 1)],
                               'Power (%)': [round(100 * power_con, 1)]})
    EC0 = ["{:.1e}".format(cost_con)]
    EC1 = ["{:.1e}".format(EC1_con)]
    EC2 = ["{:.1e}".format(EC2_con)]

    # Plot Contour Level Lines:
    # Specify Meshgrid Dimensions
#         delta_x, delta_y, x_interv = 10, 0.2, 500
    delta_x, delta_y, x_interv = 10, 4, 500
    max_n = round(n_con, 0)
#         n_x = np.arange(max(0.0, max_n - x_interv), max_n + x_interv, delta_x)
    n_x = np.arange(max(0.0, max_n - 1), max_n + 1, 1)
    lambda_y = np.arange(-1.0, 3.0, delta_y)
    n_X, lambda_Y = np.meshgrid(n_x, lambda_y)

    # Compute Expected Cost Over the Grid:
    Exp_Cost = np.empty((len(lambda_y), len(n_x))) * 0
#         for j in range(len(n_x)):
#             for i in range(len(lambda_y)):
#                 Exp_Cost[i, j] = C_n([n_X[i, j], lambda_Y[i, j]])

    dic_plt = pd.DataFrame({'n_X': np.concatenate(n_X).flat, 'lambda_Y': np.concatenate(lambda_Y).flat,
                            'Exp_Cost': np.concatenate(Exp_Cost).flat})
    dic_scatter = dict({'x': [n_unc, n_con], 'y': [lambda_unc, lambda_con], 'label': ['Unconstrained', 'Constrained']})


    return results_df, dic_scatter, dic_plt, EC0, EC1, EC2


# Set up Table Output:
BDA_res, dic_scatter, dic_plt, EC0, EC1, EC2 = Run_BDA(c1_gui.value, s2_gui.value, k_gui.value, N_gui.value,
                                                       Disease_Name_gui.value, Disease_Rank_gui, sigma_gui.value,
                                                       p_H0_gui.value, maxPower_gui.value, kappa_gui.value,
                                                       cf_gui.value)
Columns = [TableColumn(field=Ci, title=Ci) for Ci in BDA_res.columns]  # bokeh columns
Columns1 = [TableColumn(field=Ci, title=Ci) for Ci in BDA_res.columns[0:4]]  # bokeh columns
Columns2 = [TableColumn(field=Ci, title=Ci) for Ci in BDA_res.columns[4:]]  # bokeh columns
Columns3 = [TableColumn(field='Optimal Cost', title='Optimal Cost'),
            TableColumn(field='Type-I Optimal Cost', title='Type-I Optimal Cost'),
            TableColumn(field='Type-II Optimal Cost', title='Type-II Optimal Cost')]  # bokeh columns

source_df = ColumnDataSource(data=BDA_res)
source_df3 = ColumnDataSource(data={'Optimal Cost': EC0, 'Type-I Optimal Cost': EC1, 'Type-II Optimal Cost': EC2})

source_obj = DataTable(columns=Columns, source=source_df, width=1000, height=75, index_position=None)  # bokeh table
source_obj1 = DataTable(columns=Columns1, source=source_df, width=700, height=60, index_position=None)  # bokeh table
source_obj2 = DataTable(columns=Columns2, source=source_df, width=380, height=60, index_position=None)  # bokeh table
source_obj3 = DataTable(columns=Columns3, source=source_df3, width=330, height=60, index_position=None)  # bokeh table

# Set up Plot Output:
plot_opt = figure(plot_width=800, plot_height=525, tools='pan,wheel_zoom,box_select,reset, save',
                  x_axis_label='Sample Size n', y_axis_label='Critical Value \u03BB')
plot_opt.title.text = '%s (Contour Plot)' % Disease_Name_gui.value

colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
mapper = LinearColorMapper(palette=colors, low=dic_plt.Exp_Cost.min(), high=dic_plt.Exp_Cost.max())
source_plt = ColumnDataSource(data=dic_plt)
plot_opt.rect(source=source_plt, x='n_X', y='lambda_Y', width=20, height=0.2, line_color=None,
              fill_color={'field': 'Exp_Cost', 'transform': mapper})
color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="12px",
                     ticker=BasicTicker(desired_num_ticks=len(colors)),
                     formatter=PrintfTickFormatter(format="%.1e"),
                     label_standoff=13, border_line_color=None, location=(0, 0))
plot_opt.add_layout(color_bar, 'right')
plot_opt.add_layout(Legend(), 'right')

source_scatter = ColumnDataSource(data=dic_scatter)
cat_color_mapper = CategoricalColorMapper(factors=['Unconstrained', 'Constrained'], palette=[RdBu3[0], RdBu3[2]])
plot_opt.circle('x', 'y', size=20, source=source_scatter, color={'field': 'label', 'transform': cat_color_mapper},
                legend_group='label', selection_color="orange", alpha=0.6,
                nonselection_alpha=0.1, selection_alpha=0.4)

# Spinner: https://www.w3schools.com/howto/howto_css_loader.asp
spinner_text = """
<!-- https://www.w3schools.com/howto/howto_css_loader.asp -->
<div class="loader">
<style scoped>
.loader {
    border: 16px solid #f3f3f3; /* Light grey */
    border-top: 16px solid #3498db; /* Blue */
    border-radius: 50%;
    width: 120px;
    height: 120px;
    animation: spin 2s linear infinite;
    left:0;
    right:0;
    top:0;
    bottom:0;
    position:absolute;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
} 
</style>
</div>
"""
div_spinner = Div(text="", width=0, height=0)


def show_spinner1():
    div_spinner.width = 120
    div_spinner.height = 120
    div_spinner.text = spinner_text


def show_spinner2():
    div_spinner.width = 120
    div_spinner.height = 130
    div_spinner.text = spinner_text
    curdoc().add_next_tick_callback(update_data2)


def hide_spinner():
    div_spinner.width = 0
    div_spinner.height = 0
    div_spinner.text = ""
# End of Spinner

# Set up callbacks
'''
def update_data(attrname, old, new):
    # Get the current slider values
    Disease_Name = Disease_Name_gui.value
    Disease_Rank = My_Data_df.loc[Disease_Name, 'YLL Rank']

    if (old in My_Data_df.index.tolist()) or (new in My_Data_df.index.tolist()):
        N = My_Data_df.loc[Disease_Name, 'Prevalence']
        c_1 = My_Data_df.loc[Disease_Name, 'c1']
        s_2 = My_Data_df.loc[Disease_Name, 's2']
        N_gui.value = N
        c1_gui.value = c_1
        s2_gui.value = s_2

        maxPower = maxpower_0
        p_H0 = p0_0
        sigma = sigma_0
        k = k_0
        maxPower_gui.value = maxPower
        p_H0_gui.value = p_H0
        sigma_gui.value = sigma
        k_gui.value = k

    elif (old in ['Yes', 'No']) or (new in ['Yes', 'No']):
        if max_power_check_gui.value == 'No':
            maxPower_gui.value = 1.0
        if trial_cost_check_gui.value == 'No':
            kappa_gui.value = 0.001
            cf_gui.value = 0.0

        N = N_gui.value
        c_1 = c1_gui.value
        s_2 = s2_gui.value
        maxPower = maxPower_gui.value
        p_H0 = p_H0_gui.value
        sigma = sigma_gui.value
        k = k_gui.value
        if maxPower < 1.0:
            max_power_check_gui.value = 'Yes'

    else:
        N = N_gui.value
        c_1 = c1_gui.value
        s_2 = s2_gui.value
        maxPower = maxPower_gui.value
        p_H0 = p_H0_gui.value
        sigma = sigma_gui.value
        k = k_gui.value
        if maxPower < 1.0:
            max_power_check_gui.value = 'Yes'
        else:
            max_power_check_gui.value = 'No'

    kappa = kappa_gui.value
    cf = cf_gui.value

    # Generate the Results
    BDA_res = pd.DataFrame(columns=[
        'YLL Rank', 'Disease Name', 'Prevalence (Thousands)', 'Severity',
        'Sample Size', 'Critical Value', 'alpha (%)', 'Power (%)'])
    results_df, dic_scatter, dic_plt, EC0, EC1, EC2 = Run_BDA(c_1, s_2, k, N, Disease_Name, Disease_Rank,
                                                              sigma, p_H0, maxPower, kappa, cf)
    BDA_res = pd.concat([BDA_res, results_df], axis=0, join='outer', sort=False).reset_index(drop=True)

    source_df.data = BDA_res
    source_df3 = ColumnDataSource(data={'Optimal Cost': EC0, 'Type-I Optimal Cost': EC1, 'Type-II Optimal Cost': EC2})
    source_obj.source = source_df
    source_obj1.source = source_df
    source_obj2.source = source_df
    source_obj3.source = source_df3
    source_scatter.data = dic_scatter
    source_plt.data = dic_plt
    plot_opt.title.text = Disease_Name
    mapper.low = dic_plt.Exp_Cost.min()
    mapper.high = dic_plt.Exp_Cost.max()
'''


def update_data1(attrname, old, new):
    curdoc().add_next_tick_callback(show_spinner1)
    # Get the current slider values
    Disease_Name = Disease_Name_gui.value
    Disease_Rank = My_Data_df.loc[Disease_Name, 'YLL Rank']

    if (old in My_Data_df.index.tolist()) or (new in My_Data_df.index.tolist()):
        N = My_Data_df.loc[Disease_Name, 'Prevalence']
        c_1 = My_Data_df.loc[Disease_Name, 'c1']
        s_2 = My_Data_df.loc[Disease_Name, 's2']
        N_gui.value = N
        c1_gui.value = c_1
        s2_gui.value = s_2

        maxPower = maxpower_0
        p_H0 = p0_0
        sigma = sigma_0
        k = k_0
        maxPower_gui.value = maxPower
        p_H0_gui.value = p_H0
        sigma_gui.value = sigma
        k_gui.value = k
        max_power_check_gui.value = 'Yes'
        #trial_cost_check_gui.value = 'Yes'

    elif (old in ['Yes', 'No']) or (new in ['Yes', 'No']):
        if max_power_check_gui.value == 'No':
            maxPower_gui.value = 1.0
        if trial_cost_check_gui.value == 'No':
            kappa_gui.value = 0.001
            cf_gui.value = 0.0

        N = N_gui.value
        c_1 = c1_gui.value
        s_2 = s2_gui.value
        maxPower = maxPower_gui.value
        p_H0 = p_H0_gui.value
        sigma = sigma_gui.value
        k = k_gui.value
        if maxPower < 1.0:
            max_power_check_gui.value = 'Yes'

    else:
        N = N_gui.value
        c_1 = c1_gui.value
        s_2 = s2_gui.value
        maxPower = maxPower_gui.value
        p_H0 = p_H0_gui.value
        sigma = sigma_gui.value
        k = k_gui.value
        if maxPower < 1.0:
            max_power_check_gui.value = 'Yes'
        else:
            max_power_check_gui.value = 'No'

    kappa = kappa_gui.value
    cf = cf_gui.value

    # Generate the Results
    BDA_res = pd.DataFrame(columns=[
        'YLL Rank', 'Disease Name', 'Prevalence (Thousands)', 'Severity',
        'Sample Size', 'Critical Value', 'alpha (%)', 'Power (%)'])
    results_df, dic_scatter, dic_plt, EC0, EC1, EC2 = Run_BDA(c_1, s_2, k, N, Disease_Name, Disease_Rank,
                                                              sigma, p_H0, maxPower, kappa, cf)
    BDA_res = pd.concat([BDA_res, results_df], axis=0, join='outer', sort=False).reset_index(drop=True)

    source_df.data = BDA_res
    source_df3 = ColumnDataSource(data={'Optimal Cost': EC0, 'Type-I Optimal Cost': EC1, 'Type-II Optimal Cost': EC2})
    source_obj.source = source_df
    source_obj1.source = source_df
    source_obj2.source = source_df
    source_obj3.source = source_df3
    source_scatter.data = dic_scatter
    source_plt.data = dic_plt
    plot_opt.title.text = Disease_Name
    mapper.low = dic_plt.Exp_Cost.min()
    mapper.high = dic_plt.Exp_Cost.max()

    curdoc().add_next_tick_callback(hide_spinner)


def update_data2():
    #curdoc().add_next_tick_callback(show_spinner)
    # Get the current slider values
    Disease_Name = Disease_Name_gui.value
    Disease_Rank = My_Data_df.loc[Disease_Name, 'YLL Rank']

    N = N_gui.value
    c_1 = c1_gui.value
    s_2 = s2_gui.value
    maxPower = maxPower_gui.value
    p_H0 = p_H0_gui.value
    sigma = sigma_gui.value
    k = k_gui.value
    if maxPower < 1.0:
        max_power_check_gui.value = 'Yes'
    else:
        max_power_check_gui.value = 'No'

    kappa = kappa_gui.value
    cf = cf_gui.value

    # Generate the Results
    BDA_res = pd.DataFrame(columns=[
        'YLL Rank', 'Disease Name', 'Prevalence (Thousands)', 'Severity',
        'Sample Size', 'Critical Value', 'alpha (%)', 'Power (%)'])
    results_df, dic_scatter, dic_plt, EC0, EC1, EC2 = Run_BDA(c_1, s_2, k, N, Disease_Name, Disease_Rank,
                                                              sigma, p_H0, maxPower, kappa, cf)
    BDA_res = pd.concat([BDA_res, results_df], axis=0, join='outer', sort=False).reset_index(drop=True)

    source_df.data = BDA_res
    source_df3 = ColumnDataSource(data={'Optimal Cost': EC0, 'Type-I Optimal Cost': EC1, 'Type-II Optimal Cost': EC2})
    source_obj.source = source_df
    source_obj1.source = source_df
    source_obj2.source = source_df
    source_obj3.source = source_df3
    source_scatter.data = dic_scatter
    source_plt.data = dic_plt
    plot_opt.title.text = Disease_Name
    mapper.low = dic_plt.Exp_Cost.min()
    mapper.high = dic_plt.Exp_Cost.max()

    curdoc().add_next_tick_callback(hide_spinner)


# for params_gui in [Disease_Name_gui, max_power_check_gui, maxPower_gui, p_H0_gui, sigma_gui,
#                        N_gui, c1_gui, s2_gui, k_gui, trial_cost_check_gui, kappa_gui, cf_gui]:
#     params_gui.on_change('value', update_data)


# Update data from boxes
for params_gui in [Disease_Name_gui, max_power_check_gui, trial_cost_check_gui]:
    params_gui.on_change('value', update_data1)

# Update data from sliders
submit.on_click(show_spinner2)


# Set up Table Layout
Tab1_title = Div(text='<b>Model Inputs:</b>')
Tab2_title = Div(text='<b>BDA Results: (Constrained)</b>')
Tab3_title = Div(text='<b>Optimal Cost: (Constrained)</b>')

# Set up layouts and add to document
sub_widgets = row(column(Tab2_title, source_obj2), column(Tab3_title, source_obj3))
widgets = column(Tab1_title, source_obj1, sub_widgets, plot_opt)

# Input Divider:
Wid0_title = Div(text='<i>The default input values used are taken from Tables 2-3 of the paper.</i>')
Wid_submit = Div(text='<b>Click Here to Calculate the BDA Outputs:</b>')
Wid1_title = Div(text='<b>Disease:</b>')
Wid2_title = Div(text='<b>Power Constraint:</b>')
Wid3_title = Div(text='<b>BDA Inputs:</b>')
Wid4_title = Div(text='<b>Trial Costs:</b>')
inputs = column(Wid0_title, Wid_submit, submit, div_spinner, Wid1_title, Disease_Name_gui, Wid2_title, max_power_check_gui, maxPower_gui,
                Wid3_title, p_H0_gui, N_gui, c1_gui, s2_gui, k_gui,
                Wid4_title, trial_cost_check_gui, kappa_gui, cf_gui)
# GUI Title
gui_title = Div(text="<b style='color: DarkRed; font-size: 40px'>Is the FDA too conservative or too aggressive? </b>" +
                     "<a href='https://www.sciencedirect.com/science/article/pii/S0304407618302380'><i style='color: "
                     "BlueViolet; font-size: 20px'> (Isakov et al.)</i></a>" +
                     "<h1 style='color: FireBrick; font-size: 20px'>A Bayesian decision analysis of clinical trial design "
                     "</h1>")

# GUI Notes
gui_notes = Div(text="<h1 style='Black: blue; font-size: 18px'> Notes: </h1>" +
                "<p style='color: DimGrey; font-size: 10px'> <b> (1) Prevalence (N): </b>"
                "Size of patient population.</p>" +
                "<p style='color: DimGrey; font-size: 10px'> <b> (2) Type-I Cost (c\u2081): </b>"
                "Cost of side effects per patient. </p>" +
                "<p style='color: DimGrey; font-size: 10px'> <b> (3) Type-II Cost (c\u2082): </b>"
                "Burden of disease per patient. </p>" +
                "<p style='color: DimGrey; font-size: 10px'> <b> (4) Treatment Effect Scaling Parameter (n): </b>"
                "Treatment Effect = (1/2)^n x \u03C3 .</p>" +
                "<p style='color: DimGrey; font-size: 10px'> <b> (5) Severity (s\u2082): </b>"
                "s\u2082 = c\u2082 x 2^n .</p>" +
                "<p style='color: DimGrey; font-size: 10px'> <b> (6) \u03BA (in Thousands): </b>"
                "Financial dollar-loss equivalent to every year of healthy life lost to the disease or adverse effects"
                " of medical treatment for each patient." +
                "<p style='color: DimGrey; font-size: 10px'> <b> (7) Trial Costs per Patient (c_f): </b>"
                "Per-patient operational cost for running the clinical trial. </p>" +
                "<p style='color: DimGrey; font-size: 10px'> <b> (8) YLL Rank: </b>"
                "Ranking diseases based on the years of life lost due to premature death. </p>")

main_row = row(inputs, widgets)
main_col = column(gui_title, main_row, gui_notes)

curdoc().add_root(main_col)
curdoc().title = "Is the FDA too conservative or too aggressive?"

