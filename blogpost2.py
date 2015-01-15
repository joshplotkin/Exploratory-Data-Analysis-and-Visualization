import pandas as pd
import numpy as np
from pandas import *
import os
import brewer2mpl
import pylab as P
import matplotlib as plt
import matplotlib
import matplotlib.pyplot as plt
import scipy as sc
from scipy import stats
from scipy.stats import norm
from pylab import *

pd.set_option('max_rows',4000)
pd.set_option('max_columns',1000)
pandas.set_option('precision',3)

os.chdir('c:/users/jp/desktop/dropbox/school/spring2014/edav/BLOG2')

def find_dups(df, colm):
	## find if there are any players with the same name/ID (can tell if the name is the same but projected wOBA is different)
	dups = []
	for i in range(len(df[colm].unique())):
		name = df[colm].unique()[i]
		this_name = df[df[colm] == name]
		if len(this_name) > 1:
			indx = range(len(this_name))
			this_name['indx'] = range(len(this_name))
			this_name = this_name.set_index(this_name['indx'])
			for stat in ('AVG', 'OBP', 'SLG', 'wOBA'): ## check if all the stats are the same (if so, we can safely assume it's the same player)
				value = this_name[stat][0] 
				for j in range(len(this_name)):
					if value != this_name[stat][j]:
						dups.append(this_name[colm])

	all_names = [] ## make a list of unique names
	for i in dups:
		for j in i:
			if j not in all_names:
				all_names.append(j)

	return all_names


def clean_names_add_ID(df):
## first get rid of . in names, which causes problems (some sites will list a name as AJ, others A.J. for example)
## then need an ID code to identify players, rather than arbitrary indexing
	replace_dot = lambda x: x.replace('.','')
	df.Name = df.Name.apply(replace_dot)

	split_name = lambda x: x.split(' ')
	df['split'] = df.Name.apply(split_name)

	first_id = lambda x: x[0][:4].upper()
	last_id = lambda x: x[1][:4].upper()

	df['ID'] = df.split.apply(last_id) + df.split.apply(first_id)
	del df['split']

	return df

def replace_dups(df,name):

	dup_list = find_dups(df,name)

	if len(dup_list) > 0:
		dup_names = df[df[name] == dup_list[0]]
		if len(dup_list) > 1:
			for i in range(1,len(dup_list)):
				dup_names = dup_names.append(df[df[name] == dup_list[i]])
	else:
		return df, None

	dup_indexes = dup_names.index
	duplicates = dup_names.Name.unique()

	## append a digit to the end of each duplicate name to make it unique
	k = 0
	for i in dup_indexes:
		df[name][i] = df[name][i] + str(k)
		k += 1

	return df, duplicates
#################################################
#################################################
PA = read_csv('PA2.csv')
# print find_dups(PA,'Name')

## there are 2 Tony Gwynns... actually, this is a mistake in the data (there aren't really 2 different ones), so we will need to remove him.
PA = PA[PA.Name != 'Tony Gwynn']


PA = clean_names_add_ID(PA)
## find duplicates in the ID field now
# print find_dups(PA,'Name')

## there are no dupplicates, so we can now start to build the player list, indexing on the IDs

## sum up the PAs, grouping by ID... this will be the start of the core dataframe
appearances = PA.groupby('ID').sum().PA

hitters = pd.DataFrame(index = appearances.index)
hitters= hitters.join(appearances, how = 'left')


positions = ('C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH')
for p in positions:
	hitters[p] = 0

for i in range(len(appearances)):
	player_pos = PA[PA.ID == appearances.index[i]]

	pos_list = PA[PA.ID == appearances.index[i]].POS.values
	this_name = player_pos.ID.unique()

	for p in positions:
		if p in pos_list:
			hitters[p][this_name] = 1

### Add full names to dataframme
PA_name = PA.copy()
PA_name = PA_name.set_index('ID')
hitters = hitters.join(PA_name.Name, how = 'inner').drop_duplicates()


hitters = hitters[['Name', 'PA', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH']]
hitters = hitters.rename(columns = {'PA' : 'PA_Proj'})
hitters.to_csv('hitters.csv')

#################################################
#################################################


def clean_projections(df):

	# first name is not 'Name' so translate that
	df = df.rename(columns = {df.columns[0] : 'Name'})


	df = clean_names_add_ID(df)
	# find_dups(df, 'ID') ## no duplicate IDs
	temp = replace_dups(df, 'ID')
	df = temp[0]

	df = df.set_index('ID')
	del df['playerid']

	return df, temp[1]


def return_dups_df(df, dups_list):

	df = df.copy() # make sure we don't change the original data frame

	if dups_list != None:
		lamb_dup = lambda x: x in dups_list
		df['DUP'] = df.Name.apply(lamb_dup)
		dups_df = df[df.DUP == True]
		return dups_df
	else:
		return [] # return an empty list


def remove_dups_df(df):

	dups_df = return_dups_df(df[0], df[1])
	df = df[0]

	if len(dups_df) > 0: # if the list of dups isn't empty
		# remove duplicates and remove last digit from index for the duplicates we've kept
		dups_df['ID'] = dups_df.index.copy() # append a new column that is a copy of the index
		df['ID'] = df.index.copy() 
		dups_df['REMOVE'] = ''

		remove_digit = lambda x: x[:-1] if x[-1:] in ['0','1','2','3','4','5','6','7','8','9'] else x # if the index ends with an integer, we need to remove it so it is consistent with the other data frame
		for i in range(1,5): # remove the digits at the end of the IDs (loop to remove multiple numbers)
			dups_df['ID'] = dups_df.ID.apply(remove_digit)
		
		uniques = dups_df.ID.unique() # list of repeated IDs

		for u in uniques: # for each duplicate ID
			name = dups_df[dups_df.ID == u] # data frame of players sharing particular duplicate ID

			maxWAR = name.WAR.max() # we'll keep the player with the highest WAR and remove the rest
			remove = lambda x: False if x == maxWAR else True 
			name.REMOVE = name.WAR.apply(remove) 
			name = name[name.REMOVE == True]


			id_to_remove =  name.index # list of players to remove
			for i in id_to_remove: # loop through the list and remove one at a time
				df = df[df.ID != i]
		
		for i in range(1,5): # remove the digits at the end of the IDs (loop to remove multiple numbers)
			df['ID'] = df.ID.apply(remove_digit)

		df = df.set_index('ID')

	return df

####################################################################################
####################################################################################

## call functions to clean up and format the data, and find any players who have duplicate IDs
## the functions also remove the worst players who share an ID, as those players are all insignificant anyway
## note for future: always investigate the duplicate names, because in the future, there might be 
## 2 good players who share the same name. in 2014, that is not the case.

zips = remove_dups_df(clean_projections(read_csv('ZIPS-B.csv')))
fans = remove_dups_df(clean_projections(read_csv('FANS-B.csv')))
steamer = remove_dups_df(clean_projections(read_csv('STEAMER-B.csv')))
oliver = remove_dups_df(clean_projections(read_csv('OLIVER-B.csv')))


def adjust_by_PA(df, PA, site):

	### next step: join the projected PAs to the data frames
	df = df.join(PA.PA_Proj, how = 'left')

	### next step: remove all players with under 400 PAs (part-time players)
	df = df[df.PA_Proj > 400]

	### next step background: projection systems are good at projection performance, and
	### bad at projecting playing time. we'll use the PA_Proj source which is much better
	### and adjust the raw numbers to that (which excludes things like AVG)
	### since I am using plate appearances, I can only change hitting stats (no fielding or WAR)
	df_adjusted = df[['Name', 'PA_Proj', 'PA', 'H', '2B', '3B', 'HR', 'R', 'RBI', 'BB', 'SO', 'HBP', 'SB','CS']].copy()


	for stat in ('H', '2B', '3B', 'HR', 'R', 'RBI', 'BB', 'SO', 'HBP', 'SB','CS'):
		df_adjusted[stat] = df_adjusted[stat]*df.PA_Proj/df_adjusted.PA

	# get rid of what isn't needed for the merge and re-order
	order_adj = ['H', '2B', '3B', 'HR', 'R','RBI', 'BB', 'SO', 'HBP', 'SB', 'CS']
	order_reg =  ['H', '2B', '3B', 'HR', 'R','RBI', 'BB', 'SO', 'HBP', 'SB', 'CS', 'AVG', 'OBP', 'SLG', 'OPS', 'wOBA', 'Fld', 'BsR', 'WAR']

	df_adjusted = df_adjusted[order_adj]
	df = df[order_reg]

	# rename the columns so I can merge later
	for col in df_adjusted.columns:
		df_adjusted = df_adjusted.rename(columns = {col : col + '_ADJ_' + site})

	for col in df.columns:
		df = df.rename(columns = {col : col + '_' + site})

	return df, df_adjusted


PAs = read_csv('hitters.csv') # pull in the PA dataframe from before
# PAs.columns = ('Name', 'ID','PA_Proj', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH')
PAs = PAs.set_index('ID')

oliver_both = adjust_by_PA(oliver, PAs, 'O')
oliver = oliver_both[0]
oliver_adjusted = oliver_both[1]

zips_both = adjust_by_PA(zips, PAs, 'Z')
zips = zips_both[0]
zips_adjusted = zips_both[1]

steamer_both = adjust_by_PA(steamer, PAs, 'S')
steamer = steamer_both[0]
steamer_adjusted = steamer_both[1]

fans_both = adjust_by_PA(fans, PAs, 'F')
fans = fans_both[0]
fans_adjusted = fans_both[1]

#### next step : calculate Z-Scores based on the top 100 players
PAs = PAs.join(oliver, how = 'left')
PAs = PAs.join(oliver_adjusted, how = 'left')

PAs = PAs.join(fans, how = 'left')
PAs = PAs.join(fans_adjusted, how = 'left')

PAs = PAs.join(steamer, how = 'left')
PAs = PAs.join(steamer_adjusted, how = 'left')

PAs = PAs.join(zips, how = 'left')
PAs = PAs.join(zips_adjusted, how = 'left')

PAs = PAs[PAs.PA_Proj > 400]
PAs = PAs.sort('PA_Proj', ascending = False)


def rearrange_and_delete_cols(df, list): #rearrange the order of the columns
	j = 0
	for i in list:
		df.insert(j,i,df.pop(i)) # reorder
		j += 1 # increment order counter

	while len(df.columns) > j: # j represents last column we wanted
		del df[df.columns[j]] # delete all columns after j

	return df

cols = ('Name', 'PA_Proj', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH', '2B_ADJ_F', '2B_ADJ_O', '2B_ADJ_S', '2B_ADJ_Z', '2B_F', '2B_O', '2B_S', '2B_Z', '3B_ADJ_F', '3B_ADJ_O', '3B_ADJ_S', '3B_ADJ_Z', '3B_F', '3B_O', '3B_S', '3B_Z', 'AVG_F', 'AVG_O', 'AVG_S', 'AVG_Z', 'BB_ADJ_F', 'BB_ADJ_O', 'BB_ADJ_S', 'BB_ADJ_Z', 'BB_F', 'BB_O', 'BB_S', 'BB_Z', 'BsR_F', 'BsR_O', 'BsR_S', 'BsR_Z', 'CS_ADJ_F', 'CS_ADJ_O', 'CS_ADJ_S', 'CS_ADJ_Z', 'CS_F', 'CS_O', 'CS_S', 'CS_Z', 'Fld_F', 'Fld_O', 'Fld_S', 'Fld_Z', 'H_ADJ_F', 'H_ADJ_O', 'H_ADJ_S', 'H_ADJ_Z', 'H_F', 'H_O', 'H_S', 'H_Z', 'HBP_ADJ_F', 'HBP_ADJ_O', 'HBP_ADJ_S', 'HBP_ADJ_Z', 'HBP_F', 'HBP_O', 'HBP_S', 'HBP_Z', 'HR_ADJ_F', 'HR_ADJ_O', 'HR_ADJ_S', 'HR_ADJ_Z', 'HR_F', 'HR_O', 'HR_S', 'HR_Z', 'OBP_F', 'OBP_O', 'OBP_S', 'OBP_Z', 'OPS_F', 'OPS_O', 'OPS_S', 'OPS_Z', 'RBI_ADJ_F', 'RBI_ADJ_O', 'RBI_ADJ_S', 'RBI_ADJ_Z', 'RBI_F', 'RBI_O', 'RBI_S', 'RBI_Z', 'R_ADJ_F', 'R_ADJ_O', 'R_ADJ_S', 'R_ADJ_Z', 'R_F', 'R_O', 'R_S', 'R_Z', 'SB_ADJ_F', 'SB_ADJ_O', 'SB_ADJ_S', 'SB_ADJ_Z', 'SB_F', 'SB_O', 'SB_S', 'SB_Z', 'SLG_F', 'SLG_O', 'SLG_S', 'SLG_Z', 'SO_ADJ_F', 'SO_ADJ_O', 'SO_ADJ_S', 'SO_ADJ_Z', 'SO_F', 'SO_O', 'SO_S', 'SO_Z', 'WAR_F', 'WAR_O', 'WAR_S', 'WAR_Z', 'wOBA_F', 'wOBA_O', 'wOBA_S', 'wOBA_Z')

PAs = rearrange_and_delete_cols(PAs, cols)

# add ages
age = read_csv('AGES.csv')
age = age.set_index('ID')
PAs = PAs.join(age.Age, how = 'left')

# get rid of any players who don't have projections from all 4 systems
PAs = PAs[PAs.WAR_F.notnull()]
PAs = PAs[PAs.WAR_S.notnull()]
PAs = PAs[PAs.WAR_O.notnull()]
PAs = PAs[PAs.WAR_Z.notnull()]

PAs.to_csv('AGGREGATE_HITTERS.csv')
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################

### CODE I FOUND TO MAKE PLOTS LOOK LIKE ggplot2
def rstyle(ax): 
    """Styles an axes to appear like ggplot2
    Must be called after all plot and axis manipulation operations have been carried out (needs to know final tick spacing)
    """
    #set the style of the major and minor grid lines, filled blocks
    ax.grid(True, 'major', color='w', linestyle='-', linewidth=1.4)
    ax.grid(True, 'minor', color='0.92', linestyle='-', linewidth=0.7)
    ax.patch.set_facecolor('0.85')
    ax.set_axisbelow(True)
    
    #set minor tick spacing to 1/2 of the major ticks
    ax.xaxis.set_minor_locator(MultipleLocator( (plt.xticks()[0][1]-plt.xticks()[0][0]) / 2.0 ))
    ax.yaxis.set_minor_locator(MultipleLocator( (plt.yticks()[0][1]-plt.yticks()[0][0]) / 2.0 ))
    
    #remove axis border
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_alpha(0)
       
    #restyle the tick lines
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markersize(5)
        line.set_color("gray")
        line.set_markeredgewidth(1.4)
    
    #remove the minor tick lines    
    for line in ax.xaxis.get_ticklines(minor=True) + ax.yaxis.get_ticklines(minor=True):
        line.set_markersize(0)
    
    #only show bottom left ticks, pointing out of axis
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    
    if ax.legend_ <> None:
        lg = ax.legend_
        lg.get_frame().set_linewidth(0)
        lg.get_frame().set_alpha(0.5)
        
        
def rhist(ax, data, **keywords):
    """Creates a histogram with default style parameters to look like ggplot2
    Is equivalent to calling ax.hist and accepts the same keyword parameters.
    If style parameters are explicitly defined, they will not be overwritten
    """
    
    defaults = {
                'facecolor' : '0.3',
                'edgecolor' : '0.28',
                'linewidth' : '1',
                'bins' : 100
                }
    
    for k, v in defaults.items():
        if k not in keywords: keywords[k] = v
    
    return ax.hist(data, **keywords)


def rbox(ax, data, **keywords):
    """Creates a ggplot2 style boxplot, is eqivalent to calling ax.boxplot with the following additions:
    
    Keyword arguments:
    colors -- array-like collection of colours for box fills
    names -- array-like collection of box names which are passed on as tick labels

    """

    hasColors = 'colors' in keywords
    if hasColors:
        colors = keywords['colors']
        keywords.pop('colors')
        
    if 'names' in keywords:
        ax.tickNames = plt.setp(ax, xticklabels=keywords['names'] )
        keywords.pop('names')
    
    bp = ax.boxplot(data, **keywords)
    pylab.setp(bp['boxes'], color='black')
    pylab.setp(bp['whiskers'], color='black', linestyle = 'solid')
    pylab.setp(bp['fliers'], color='black', alpha = 0.9, marker= 'o', markersize = 3)
    pylab.setp(bp['medians'], color='black')
    
    numBoxes = len(data)
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
          boxX.append(box.get_xdata()[j])
          boxY.append(box.get_ydata()[j])
        boxCoords = zip(boxX,boxY)
        
        if hasColors:
            boxPolygon = Polygon(boxCoords, facecolor = colors[i % len(colors)])
        else:
            boxPolygon = Polygon(boxCoords, facecolor = '0.95')
            
        ax.add_patch(boxPolygon)
    return bp
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
def plot_scatter_regression(hitters, xlab, ylab, title):


	## ADD JITTER:
	jitter = lambda x: np.random.uniform(-0.25,0.25) + x.astype(float)
	hitters['JITTER_AGE'] = hitters.Age.apply(jitter)

	set2 = brewer2mpl.get_map('Set1', 'qualitative', 4).mpl_colors
	fig = plt.figure()
	ax = fig.add_subplot(111)
	x = np.arange(20,42,1)

	scatter(hitters.JITTER_AGE, hitters.WAR_Z, color = set2[0], label = 'ZiPS', alpha = 0.5)
	b3,b2,b1,b0 = polyfit(hitters.Age, hitters.WAR_Z, 3) 
	plot(x, b3*x**3+b2*x**2+b1*x+b0, color = set2[0], label = 'ZiPS Reg', linewidth = 3)

	scatter(hitters.JITTER_AGE, hitters.WAR_F, color = set2[1], label = 'Fans', alpha = 0.5)
	b3,b2,b1,b0 = polyfit(hitters[hitters.WAR_F.notnull()].Age, hitters[hitters.WAR_F.notnull()].WAR_F, 3) 
	plot(x, b3*x**3+b2*x**2+b1*x+b0, color = set2[1], label = 'Fans Reg', linewidth = 3)

	scatter(hitters.JITTER_AGE, hitters.WAR_O, color = set2[2], label = 'Oliver', alpha = 0.5)
	b3,b2,b1,b0 = polyfit(hitters.Age, hitters.WAR_O, 3) 
	plot(x, b3*x**3+b2*x**2+b1*x+b0, color = set2[2], label = 'Oliver Reg', linewidth = 3)

	scatter(hitters.JITTER_AGE, hitters.WAR_S, color = set2[3], label = 'Steamer', alpha = 0.5)
	b3,b2,b1,b0 = polyfit(hitters.Age, hitters.WAR_S, 3) 
	plot(x, b3*x**3+b2*x**2+b1*x+b0, color = set2[3], label = 'Steamer Reg', linewidth = 3)

	# annotate('The Walking Outlier\n (aka Mike Trout)', xy=(22.5,9), xytext=(24.5,8.75),
 #            arrowprops=dict(facecolor='black', shrink=0.05))

	for label in ax.get_xticklines():
	    label.set_visible(False)

	for label in ax.get_yticklines():
	    label.set_visible(False)

	suptitle(title, fontsize=14, fontweight='bold')
	xlabel(xlab,fontweight='bold')
	ylabel(ylab,fontweight='bold')
	plt.legend(loc = 'best', fontsize=10)
	xticks()
	yticks()
	ylim([-2,10])
	xlim([20,42])
	rstyle(ax)
	# plt.show()
#####################################################

hitters = read_csv('AGGREGATE_HITTERS.csv')
hitters = hitters.set_index('ID')
hitters = hitters[['Age', 'Name', 'WAR_F', 'WAR_O', 'WAR_Z', 'WAR_S']]


## 02 - 3rd degree polynomial ############################################
plot_scatter_regression(hitters, 'Age', 'WAR', 'WAR vs. Age')
##########################################################################

## 03 - no Trout #########################################################
# what if Mike Trout didn't exist?
plot_scatter_regression(hitters[hitters.Name != 'Mike Trout'], 'Age', 'WAR', 'WAR vs. Age \n Excluding Mike Trout')
##########################################################################

## 04 - WAR Z-Scores - With vs. Without Trout ############################

## add columns of WAR Z-Scores
war_list = ('WAR_F', 'WAR_S', 'WAR_O', 'WAR_Z')
for c in hitters.columns:
	if c in war_list:
		hitters[c+'(Z)'] = (hitters[c] - hitters[c].mean())/hitters[c].std()

hitters['AVG'] = (hitters['WAR_F(Z)']+hitters['WAR_Z(Z)']+hitters['WAR_O(Z)']+hitters['WAR_S(Z)'])/4



set2 = brewer2mpl.get_map('Set1', 'qualitative', 4).mpl_colors

fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(20,42,1)

scatter(hitters[hitters.Name != 'Mike Trout'].Age, hitters[hitters.Name != 'Mike Trout'].AVG, color = set2[0], label = 'Not Trout', alpha = 0.5)
b5,b4,b3,b2,b1,b0 = polyfit(hitters[hitters.Name != 'Mike Trout'].Age, hitters[hitters.Name != 'Mike Trout'].AVG, 5) 
plot(x, b5*x**5+b4*x**4+b3*x**3+b2*x**2+b1*x+b0, color = set2[0], label = 'Excluding Trout', linewidth = 3)

scatter(hitters[hitters.Name == 'Mike Trout'].Age, hitters[hitters.Name == 'Mike Trout'].AVG, color = set2[1], label = 'Trout', alpha = 0.5)
b5,b4,b3,b2,b1,b0 = polyfit(hitters.Age, hitters.AVG, 5) 
plot(x, b5*x**5+b4*x**4+b3*x**3+b2*x**2+b1*x+b0, color = set2[1], label = 'Including Trout', linewidth = 3)

for label in ax.get_xticklines():
    label.set_visible(False)
for label in ax.get_yticklines():
    label.set_visible(False)

suptitle('Age vs. WAR Regression Curves\n With and Without Mike Trout', fontsize=14, fontweight='bold')
xlabel('Age',fontweight='bold')
ylabel('WAR',fontweight='bold')
plt.legend(loc = 'best', fontsize=10)
xticks()
yticks()
ylim([-3,4.5])
xlim([20,42])
rstyle(ax)
# plt.show()

##########################################################################

## 05 - CUMULATIVE WAR FOR YOUNG PLAYERS #################################

hitters['TOT_WAR'] = hitters.WAR_O + hitters.WAR_Z + hitters.WAR_S + hitters.WAR_F
hitters = hitters.sort('TOT_WAR')
## CREATE NAME ARRAY
names = []
for i in hitters[hitters.Age <23].index:
	names.append(hitters[hitters.Age <23].Name.ix[i] + '\n (Age' + str(hitters[hitters.Age <23].Age.ix[i].astype(int)) + ')')

ind = np.arange(1,len(names)+1)

width = .75

fig = plt.figure()
ax = fig.add_subplot(111)

set2 = brewer2mpl.get_map('Set1', 'qualitative', 4).mpl_colors

barh(ind,hitters[hitters.Age <23]['WAR_O']/4,width, color = set2[0], label = 'Oliver')
barh(ind,hitters[hitters.Age <23]['WAR_Z']/4,width, color = set2[1], label = 'ZiPS', left = hitters[hitters.Age <23]['WAR_O']/4)
barh(ind,hitters[hitters.Age <23]['WAR_S']/4,width, color = set2[2], label = 'Steamer', left = (hitters[hitters.Age <23]['WAR_O']+hitters[hitters.Age <23]['WAR_Z'])/4)
barh(ind,hitters[hitters.Age <23]['WAR_F']/4,width, color = set2[3], label = 'Fans', left = (hitters[hitters.Age <23]['WAR_O']+hitters[hitters.Age <23]['WAR_Z']+hitters[hitters.Age <23]['WAR_S'])/4)

suptitle('Players Under 23 Years-Old By WAR', fontsize=14, fontweight='bold')
ax.legend(loc = 'best', fontsize = 14)
xlabel('WAR',fontweight='bold')
ylim([width,ind.max()+width])
yticks(ind+width/2, names, rotation = 0, fontsize = 8)
for label in ax.get_xticklines():
    label.set_visible(False)
for label in ax.get_yticklines():
    label.set_visible(False)

rstyle(ax)

# plt.show()
##########################################################################

## 06 - YOUNG PLAYERS UNDER 24 #################################
## CREATE NAME ARRAY
names = []
for i in hitters[hitters.Age <24].index:
	names.append(hitters[hitters.Age <24].Name.ix[i] + '\n (Age' + str(hitters[hitters.Age <24].Age.ix[i].astype(int)) + ')')

ind = np.arange(1,len(names)+1)

width = .75

fig = plt.figure( figsize = (12,9))
ax = fig.add_subplot(111)

set2 = brewer2mpl.get_map('Set1', 'qualitative', 4).mpl_colors

barh(ind,hitters[hitters.Age <24]['WAR_O']/4,width, color = set2[0], label = 'Oliver')
barh(ind,hitters[hitters.Age <24]['WAR_Z']/4,width, color = set2[1], label = 'ZiPS', left = hitters[hitters.Age <24]['WAR_O']/4)
barh(ind,hitters[hitters.Age <24]['WAR_S']/4,width, color = set2[2], label = 'Steamer', left = (hitters[hitters.Age <24]['WAR_O']+hitters[hitters.Age <24]['WAR_Z'])/4)
barh(ind,hitters[hitters.Age <24]['WAR_F']/4,width, color = set2[3], label = 'Fans', left = (hitters[hitters.Age <24]['WAR_O']+hitters[hitters.Age <24]['WAR_Z']+hitters[hitters.Age <24]['WAR_S'])/4)

for label in ax.get_xticklines():
    label.set_visible(False)
for label in ax.get_yticklines():
    label.set_visible(False)

ax.invert_yaxis()
xlabel('WAR',fontweight='bold')
ylim([width,ind.max()+width])
yticks(ind+width/2, names, rotation = 0, fontsize = 8)

ax.legend(loc = 'best', fontsize = 14)
suptitle('Players Under 24 Years Old By WAR', fontsize=14, fontweight='bold')

for label in ax.get_xticklines():
    label.set_visible(False)
for label in ax.get_yticklines():
    label.set_visible(False)


rstyle(ax)

# plt.show()
##########################################################################
## 07 - AGE vs. PROSPECT STATUS ##########################################
## ADD JITTER:
jitter = lambda x: np.random.uniform(-0.1,0.1) + x.astype(float)
hitters['JITTER_AGE'] = hitters.Age.apply(jitter)

prospect = read_csv('PROSPECT.csv')
prospect = prospect.set_index('ID')
hitters = hitters.join(prospect, how = 'left')

set2 = brewer2mpl.get_map('Set1', 'qualitative', 4).mpl_colors

fig = plt.figure()
ax = fig.add_subplot(111)

x = np.arange(21,24)

scatter(hitters[hitters.Age == 21].JITTER_AGE, hitters[hitters.Age == 21].PROSPECT, color = set2[0], alpha = 1, s = 50)
scatter(hitters[hitters.Age == 22].JITTER_AGE, hitters[hitters.Age == 22].PROSPECT, color = set2[1], alpha = 1, s = 50)
scatter(hitters[hitters.Age == 23].JITTER_AGE, hitters[hitters.Age == 23].PROSPECT, color = set2[2], alpha = 1, s = 50)
b2,b1,b0 = polyfit(hitters[hitters.Age < 24].Age, hitters[hitters.Age < 24].PROSPECT, 2) 

plot(x, b2*x**2+b1*x+b0, color = set2[3], label = 'Regression (2nd Order)', linewidth = 3, marker = 'o', markersize=10, mec = set2[3])


suptitle('Age vs. Max Prospect Ranking \n (age jitter)', fontsize=14, fontweight='bold')
xlabel('Age',fontweight='bold')
ylabel('Prospect Ranking',fontweight='bold')
ylim([-2,105])
xlim([20.75,23.25])
xticks(np.arange(21,24))
yticks(np.arange(1,102,20))
ax.invert_yaxis()
annotate('Mike Trout', xy = (22.05,2))
annotate('Wil Myers', xy = (23,2))

for label in ax.get_xticklines():
    label.set_visible(False)
for label in ax.get_yticklines():
    label.set_visible(False)


rstyle(ax)
# plt.show()
##########################################################################
hitters = read_csv('AGGREGATE_HITTERS.csv')
hitters = hitters.set_index('ID')

## 08 - AVERAGE STATS BY SYSTEM ##########################################

def stat_averages(hitters, stat, adj):
	if adj == True:
		R_ADJ_Z_sum = hitters[stat+'_ADJ_Z'].mean()
		R_ADJ_S_sum = hitters[stat+'_ADJ_S'].mean()
		R_ADJ_O_sum = hitters[stat+'_ADJ_O'].mean()
		R_ADJ_F_sum = hitters[stat+'_ADJ_F'].mean()
	else: 
		R_ADJ_Z_sum = hitters[stat+'_Z'].mean()
		R_ADJ_S_sum = hitters[stat+'_S'].mean()
		R_ADJ_O_sum = hitters[stat+'_O'].mean()
		R_ADJ_F_sum = hitters[stat+'_F'].mean()

	mean = np.mean([R_ADJ_Z_sum,R_ADJ_S_sum,R_ADJ_O_sum,R_ADJ_F_sum])
	std = np.std([R_ADJ_Z_sum,R_ADJ_S_sum,R_ADJ_O_sum,R_ADJ_F_sum])

	R_ADJ_Z_sum = (R_ADJ_Z_sum - mean)/std
	R_ADJ_S_sum = (R_ADJ_S_sum - mean)/std
	R_ADJ_O_sum = (R_ADJ_O_sum - mean)/std
	R_ADJ_F_sum = (R_ADJ_F_sum - mean)/std
	return R_ADJ_Z_sum, R_ADJ_S_sum, R_ADJ_O_sum, R_ADJ_F_sum


set2 = brewer2mpl.get_map('BrBG', 'diverging', 5).mpl_colors

R_z = stat_averages(hitters, 'R', True)
wOBA_z = stat_averages(hitters, 'wOBA', False)
HR_z = stat_averages(hitters, 'HR', False)

N = 4
ind = np.arange(N)

width = 0.25

fig = plt.figure()
ax = fig.add_subplot(111)

barh([0,1,2,3],R_z,width, color = set2[0], label = 'Runs Scores')
barh([0+width,1+width,2+width,3+width],HR_z,width, color = set2[1], label = 'HR')
barh([0+2*width,1+2*width,2+2*width,3+2*width],wOBA_z,width, color = set2[4], label = 'wOBA')
yticks(ind+.4, ('ZiPS','Steamer','Oliver','Fans'), fontweight = 'bold', fontsize = 12)
xlim([-1.5,2.25])
ylim([-.25,len(ind)])
suptitle('Z-Scores vs. Average Projection System \n For Various Offensive Metrics', fontsize=14, fontweight='bold')
xlabel('Z-Score', fontweight = 'bold', fontsize = 12)


ax.legend(loc = 'best')

for label in ax.get_xticklines():
    label.set_visible(False)
for label in ax.get_yticklines():
    label.set_visible(False)

rstyle(ax)

# plt.show()
##########################################################################

## 09 - AVERAGE STATS BY SYSTEM - TOP 10 #################################
def stat_averages_10(hitters, stat, adj):
	if adj == True:
		hitters = hitters.sort(stat+'_ADJ_Z', ascending = False)
		R_ADJ_Z_sum = hitters.ix[0:11][stat+'_ADJ_Z'].mean()

		hitters = hitters.sort(stat+'_ADJ_S', ascending = False)
		R_ADJ_S_sum = hitters.ix[0:11][stat+'_ADJ_S'].mean()

		hitters = hitters.sort(stat+'_ADJ_O', ascending = False)
		R_ADJ_O_sum = hitters.ix[0:11][stat+'_ADJ_O'].mean()

		hitters = hitters.sort(stat+'_ADJ_F', ascending = False)
		R_ADJ_F_sum = hitters.ix[0:11][stat+'_ADJ_F'].mean()
	else: 
		hitters = hitters.sort(stat+'_Z', ascending = False)
		R_ADJ_Z_sum = hitters.ix[0:11][stat+'_Z'].mean()

		hitters = hitters.sort(stat+'_S', ascending = False)
		R_ADJ_S_sum = hitters.ix[0:11][stat+'_S'].mean()

		hitters = hitters.sort(stat+'_O', ascending = False)
		R_ADJ_O_sum = hitters.ix[0:11][stat+'_O'].mean()

		hitters = hitters.sort(stat+'_F', ascending = False)
		R_ADJ_F_sum = hitters.ix[0:11][stat+'_F'].mean()

	mean = np.mean([R_ADJ_Z_sum,R_ADJ_S_sum,R_ADJ_O_sum,R_ADJ_F_sum])
	std = np.std([R_ADJ_Z_sum,R_ADJ_S_sum,R_ADJ_O_sum,R_ADJ_F_sum])

	R_ADJ_Z_sum = (R_ADJ_Z_sum - mean)/std
	R_ADJ_S_sum = (R_ADJ_S_sum - mean)/std
	R_ADJ_O_sum = (R_ADJ_O_sum - mean)/std
	R_ADJ_F_sum = (R_ADJ_F_sum - mean)/std
	return R_ADJ_Z_sum, R_ADJ_S_sum, R_ADJ_O_sum, R_ADJ_F_sum

R_z = stat_averages_10(hitters, 'R', True)
wOBA_z = stat_averages_10(hitters, 'wOBA', False)
HR_z = stat_averages_10(hitters, 'HR', False)

N = 4
ind = np.arange(N)

width = 0.25

fig = plt.figure()
ax = fig.add_subplot(111)

barh([0,1,2,3],R_z,width, color = set2[0], label = 'Runs Scores')
barh([0+width,1+width,2+width,3+width],HR_z,width, color = set2[1], label = 'HR')
barh([0+2*width,1+2*width,2+2*width,3+2*width],wOBA_z,width, color = set2[4], label = 'wOBA')
yticks(ind+.4, ('ZiPS','Steamer','Oliver','Fans'), fontweight = 'bold', fontsize = 12)
xlim([-1.5,2.25])
ylim([-.25,len(ind)])
suptitle('Average Z-Scores vs. Average Projection System \n Top 10 Players at Various Offensive Metrics', fontsize=14, fontweight='bold')
xlabel('Z-Score', fontweight = 'bold', fontsize = 12)


ax.legend(loc = 'best')

for label in ax.get_xticklines():
    label.set_visible(False)
for label in ax.get_yticklines():
    label.set_visible(False)

rstyle(ax)

# plt.show()
##########################################################################

### by position?
## categories: HR, R, AVG, OBP, SLG, OPS, WAR, Fld, BsR, SO, wOBA
def plot_histo(df, stat, proj, adj):
	if adj == True:
		stat_name = stat + '_ADJ_' + proj[0:1]
	else:
		stat_name = stat + '_' + proj[0:1]

	mean = df[stat_name].mean()
	std = df[stat_name].std()


	min = mean - 3*std
	max = mean + 3*std

	# x = np.arange(min*0.8, max*1.2, abs(min-max)/100)
	x = np.arange(min,max,abs(min-max)/10000)

	fig = plt.figure()
	ax = fig.add_subplot(111)

	fill(x,sc.stats.norm.pdf(x, mean, std), color = set2[3], alpha = 1)
	hist(hitters[stat_name].dropna(), bins = 40, normed = True, histtype = 'stepfilled', cumulative=False, color = set2[2], alpha = .5)
	suptitle(proj + ' ' + stat, fontsize=14, fontweight='bold')
	xlabel(stat,fontweight='bold')
	ylabel('Frequency',fontweight='bold')

	for label in ax.get_xticklines():
	    label.set_visible(False)
	for label in ax.get_yticklines():
	    label.set_visible(False)

	rstyle(ax)

	# plt.show()

set2 = brewer2mpl.get_map('BrBG', 'diverging', 4).mpl_colors
plot_histo(hitters, 'wOBA','ZiPS',False)

# plt.show()
def plot_all_histo(df, stat, adj):

	proj = ['ZiPS','Fans','Steamer','Oliver']
	set2 = brewer2mpl.get_map('BrBG', 'diverging', 4).mpl_colors

	stat_name = []
	mean = []
	std = []

	for p in range(len(proj)):
		if adj == True:
			stat_name.append(stat + '_ADJ_' + proj[p][0:1])
		else:
			stat_name.append(stat + '_' + proj[p][0:1])

		mean.append(df[stat_name[p]].mean())
		std.append(df[stat_name[p]].std())



	min = np.mean(mean) - 4*np.mean(std)
	max = np.mean(mean) + 4*np.mean(std)

	x = np.arange(min,max,abs(min-max)/10000)

	fig = plt.figure()
	ax = fig.add_subplot(111)

	y = sc.stats.norm.pdf(x, mean[0], std[0])
	

	fill(x, sc.stats.norm.pdf(x, mean[0], std[0]), color = set2[0], alpha = 1, label = proj[0])
	fill(x, sc.stats.norm.pdf(x, mean[1], std[1]), color = set2[3], alpha = .8, label = proj[1])
	fill(x, sc.stats.norm.pdf(x, mean[2], std[2]), color = set2[1], alpha = .6, label = proj[2])
	fill(x, sc.stats.norm.pdf(x, mean[3], std[3]), color = set2[2], alpha = .4, label = proj[3])

	# hist(hitters[stat_name].dropna(), bins = 40, normed = True, histtype = 'stepfilled', cumulative=False, color = set2[2], alpha = .5)
	suptitle('All Projection Systems for ' + stat, fontsize=14, fontweight='bold')
	xlabel(stat,fontweight='bold')
	ylabel('Frequency',fontweight='bold')
	plt.legend(loc = 'best')
	xticks()
	yticks()

	for label in ax.get_xticklines():
	    label.set_visible(False)
	for label in ax.get_yticklines():
	    label.set_visible(False)


	rstyle(ax)

	# plt.show()

plot_all_histo(hitters, 'wOBA', False)
# plt.show()

## plot all WAR for all players

fig = plt.figure()
ax = fig.add_subplot(111)
hist(hitters.wOBA_Z.dropna(), bins = 40, histtype = 'stepfilled', label = 'ZiPS', normed = True, cumulative=False, alpha = 1, color = set2[0])
hist(hitters.wOBA_F.dropna(), bins = 40, histtype = 'stepfilled', label = 'Fans', normed = True, cumulative=False, alpha = 1, color = set2[3])
hist(hitters.wOBA_S.dropna(), bins = 40, histtype = 'stepfilled', label = 'Steamer', normed = True, cumulative=False, alpha = 0.65, color = set2[1])
hist(hitters.wOBA_O.dropna(), bins = 40, histtype = 'stepfilled', label = 'Oliver', normed = True, cumulative=False, alpha = 0.5, color = set2[2])

suptitle('wOBA by Projection System', fontsize=14, fontweight='bold')
xlabel('wOBA',fontweight='bold')
ylabel('Frequency',fontweight='bold')
ax.legend(loc = 'best')

# ylim([0,0.05])
rstyle(ax)

for label in ax.get_xticklines():
    label.set_visible(False)
for label in ax.get_yticklines():
    label.set_visible(False)

# plt.show()

#####

plt.hist([hitters.wOBA_Z.dropna(),hitters.wOBA_F.dropna(),hitters.wOBA_O.dropna(),hitters.wOBA_S.dropna()], bins = 10, histtype = 'bar', label = ['ZiPS','Fans','Oliver','Steamer'], normed = True, cumulative=False)#, color = set2[0])
plt.suptitle('Histogram of WAR', fontsize=14, fontweight='bold')
plt.xlabel('Wins Above Replacement',fontweight='bold')
plt.ylabel('Frequency',fontweight='bold')
plt.legend(loc = 'best')
rstyle(ax)

# plt.show()
##########################################################################

## GAUSSIAN PLOT WITH Z-SCORES ###########################################

# MAKE A DATAFRAME OF ZSCORES
hitters_zscore = hitters[hitters.Age < 40].copy()
for i in hitters.columns[11:-1]:
	hitters_zscore[i] = (hitters[i]-hitters[i].mean())/hitters[i].std()
WAR_Zscore = hitters_zscore[['Name','Age','WAR_O','WAR_Z','WAR_F','WAR_S']].dropna()
WAR_Zscore['SUM'] =  (WAR_Zscore.WAR_O+WAR_Zscore.WAR_Z+WAR_Zscore.WAR_F+WAR_Zscore.WAR_S)/4
WAR_Zscore['STD'] = WAR_Zscore[['WAR_O','WAR_Z','WAR_F','WAR_S']].std(axis=1)
## % of players at each age
## % of WAR at each age


## ADD JITTER:
jitter = lambda x: np.random.uniform(-0.25,0.25) + x.astype(float)
WAR_Zscore['JITTER_AGE'] = hitters.Age.apply(jitter)

set2 = brewer2mpl.get_map('BrBG', 'diverging', 8).mpl_colors
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(20,42,1)

scatter(WAR_Zscore.JITTER_AGE, WAR_Zscore.STD, color = set2[6], label = 'WAR', alpha = 0.5)
b3,b2,b1,b0 = polyfit(WAR_Zscore.Age, WAR_Zscore.STD, 3) 
plot(x, b3*x**3+b2*x**2+b1*x+b0, color = set2[7], label = 'WAR', linewidth = 3)

# hist(hitters[stat_name].dropna(), bins = 40, normed = True, histtype = 'stepfilled', cumulative=False, color = set2[2], alpha = .5)
suptitle('Age vs. WAR Standard Deviation', fontsize=14, fontweight='bold')
xlabel('Age',fontweight='bold')
ylabel('WAR - Standard Deviaion',fontweight='bold')
xticks()
yticks()
ylim([0,1])
xlim([WAR_Zscore.Age.min()-0.5,WAR_Zscore.Age.max()+0.5])

rstyle(ax)

# plt.show()

#################################################################################
## CREATE NAME ARRAY

WAR_Zscore_10 = WAR_Zscore.copy()
WAR_Zscore_10 = WAR_Zscore_10.sort('STD', ascending = False)
WAR_Zscore_10 = WAR_Zscore_10.ix[:11]

print WAR_Zscore_10

names = []
for i in WAR_Zscore_10.index:
	names.append(WAR_Zscore_10.Name.ix[i] + '\n (Age' + str(WAR_Zscore_10.Age.ix[i].astype(int)) + ')')

print names

ind = np.arange(1,len(names)+1)

width = .75

fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111)

set2 = brewer2mpl.get_map('BrBG', 'diverging', 4).mpl_colors

barh(ind,WAR_Zscore_10.STD/4,width, color = set2[3], label = 'Oliver')


xlabel('Standard Deviation',fontweight='bold')
ylim([width,ind.max()+width])
yticks(ind+width/2, names, rotation = 0, fontsize = 8)
ax.invert_yaxis()

suptitle('10 Players With the Highest WAR Standard Deviation', fontsize=14, fontweight='bold')

for label in ax.get_xticklines():
    label.set_visible(False)
for label in ax.get_yticklines():
    label.set_visible(False)


rstyle(ax)

# plt.show()
##########################################################################
## CREATE NAME ARRAY

WAR_Zscore_10b = WAR_Zscore.copy()
WAR_Zscore_10b = WAR_Zscore_10b.sort('STD', ascending = True)
WAR_Zscore_10b = WAR_Zscore_10b.ix[:11]


names = []
for i in WAR_Zscore_10b.index:
	names.append(WAR_Zscore_10b.Name.ix[i] + '\n (Age' + str(WAR_Zscore_10b.Age.ix[i].astype(int)) + ')')

print names

ind = np.arange(1,len(names)+1)

width = .75

fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111)

set2 = brewer2mpl.get_map('BrBG', 'diverging', 4).mpl_colors

barh(ind,WAR_Zscore_10b.STD/4,width, color = set2[0], label = 'Oliver')


xlabel('WAR - Standard Deviation',fontweight='bold')
ylim([width,ind.max()+width])
yticks(ind+width/2, names, rotation = 0, fontsize = 8)
ax.invert_yaxis()

suptitle('10 Players with the Lowest WAR Standard Deviation', fontsize=14, fontweight='bold')

for label in ax.get_xticklines():
    label.set_visible(False)
for label in ax.get_yticklines():
    label.set_visible(False)


rstyle(ax)

# plt.show()
#################################################################################
## AGE SCATTER vs. HISTOGRAM
# MAKE A DATAFRAME OF ZSCORES
hitters = read_csv('AGGREGATE_HITTERS.csv')
hitters = hitters.set_index('ID')
hitters_zscore = hitters.copy()
for i in hitters.columns[11:-1]:
	hitters_zscore[i] = (hitters[i]-hitters[i].mean())/hitters[i].std()
WAR_Zscore = hitters_zscore[['Name','Age','WAR_O','WAR_Z','WAR_F','WAR_S']].dropna()
WAR_Zscore['SUM'] =  (WAR_Zscore.WAR_O+WAR_Zscore.WAR_Z+WAR_Zscore.WAR_F+WAR_Zscore.WAR_S)/4
WAR_Zscore['STD'] = WAR_Zscore[['WAR_O','WAR_Z','WAR_F','WAR_S']].std(axis=1)

min = WAR_Zscore.Age.min()
max = WAR_Zscore.Age.max()

set2 = brewer2mpl.get_map('BrBG', 'diverging', 4).mpl_colors

mean = hitters.Age.mean()
std = hitters.Age.std()

min = mean - 3*std
max = mean + 3*std

x = np.arange(min,max,abs(min-max)/10000)

fig = plt.figure()
ax = fig.add_subplot(111)

bin_width = (max-min)-1

fill(x,sc.stats.norm.pdf(x, mean, std), color = set2[3], alpha = 1, label = 'MLE Gaussian')
hist(WAR_Zscore.Age, bins = 20, normed = True, histtype = 'stepfilled', cumulative=False, color = set2[2], alpha = .5, label = 'Players Projected > 400 PA')
suptitle('Player Age Breakdown', fontsize=14, fontweight='bold')
xlabel('Age',fontweight='bold')
ylabel('Frequency',fontweight='bold')
ax.legend(loc='best', fontsize = 12)
xlim([20,42])

rstyle(ax)

# plt.show()

#################################################################################

#################################################################################
player_ages = WAR_Zscore.groupby('Age').mean()/4


ind = player_ages.index.values.astype(int)

width = 1

fig = plt.figure()
ax = fig.add_subplot(111)

set2 = brewer2mpl.get_map('Set1', 'qualitative', 4).mpl_colors

barh(ind,player_ages.SUM,width, color = set2[1])

suptitle('Average WAR (Z-Scores) By Age', fontsize=14, fontweight='bold')

xlabel('WAR (Z-Scores)',fontweight='bold')
ylabel('Age',fontweight='bold')
ylim([20,42])
ax.invert_yaxis()
xlim([-0.5,.3])


for label in ax.get_xticklines():
    label.set_visible(False)
for label in ax.get_yticklines():
    label.set_visible(False)


rstyle(ax)
# plt.show()
#################################################################################
#################################################################################
player_ages2 = WAR_Zscore.groupby('Age').sum()
counts = WAR_Zscore.Age.value_counts().astype(float)
player_ages2['Total_number'] = counts
player_ages2['Number_prop'] = counts/counts.sum()

player_ages2['SUM_prop'] =  - player_ages2.SUM.min() + player_ages2.SUM
player_ages2.SUM_prop = player_ages2.SUM_prop/player_ages2.SUM_prop.sum()
player_ages2.SUM /= 4 

ind = player_ages2.index.values.astype(int)

width = .45

fig = plt.figure(figsize = (12,9))
ax = fig.add_subplot(111)

set2 = brewer2mpl.get_map('Set1', 'qualitative', 3).mpl_colors


bar(ind,player_ages2.SUM_prop,width, color = set2[2], label = 'Proportion of Total WAR')
bar(ind+width,player_ages2.Number_prop,width, color = set2[1], label = 'Proportion of Total Players')
ylabel('(Bars) Proportion of All Players')
xlim([21,42])
xticks(ind+.5,ind)

for label in ax.get_xticklines():
    label.set_visible(False)

for label in ax.get_yticklines():
    label.set_visible(False)


ax2 = ax.twinx()
plot(ind,(player_ages2.SUM/player_ages2.Total_number), c = set2[0], linewidth = 3, linestyle = 'steps')
ylabel('(Line) Average War Per Player (Z-Scores)')

for label in ax2.get_yticklines():
    label.set_visible(False)

suptitle('Bars - Proportion of Tot. WAR and Total Players By Age \n Line - Average WAR per Player', fontsize=14, fontweight='bold')
ax.legend(loc = 'best', fontsize = 14)

rstyle(ax)

# plt.show()
####################################################################

## TOP 10 BY WAR (labeled) #########################################
top10 = WAR_Zscore.copy()
top10['RANK'] = top10.SUM.rank(ascending = False)

top10 = top10[top10.RANK<11].sort('RANK')

last = lambda x: x.split(' ')[1]
last_name = top10.Name.apply(last)


ind = np.arange(1,11)

width = .7

fig = plt.figure()
ax = fig.add_subplot(111)

set2 = brewer2mpl.get_map('BrBG', 'diverging', 8).mpl_colors

# bar(ind,player_ages2.SUM,width, color = set2[0], label = 'Proportion of Total WAR')
bar(ind,top10.WAR_O,width, color = set2[0], label = 'Oliver')
bar(ind,top10.WAR_Z,width, color = set2[7], label = 'ZiPS', bottom = top10.WAR_O)
bar(ind,top10.WAR_S,width, color = set2[1], label = 'Steamer', bottom = top10.WAR_O+top10.WAR_Z)
bar(ind,top10.WAR_F,width, color = set2[6], label = 'Fans', bottom = top10.WAR_O+top10.WAR_Z+top10.WAR_S)

suptitle('Top 10 Players By Average Projected WAR Z-Scores', fontsize=14, fontweight='bold')
ax.legend(loc = 'best', fontsize = 14)
ylabel('Total WAR (Z-Scores)')
xlim([1,10.75])
ax.tick_params(axis = 'x', which = 'both', bottom = 'off')
xticks(ind+width/2,last_name, rotation = 30, fontsize = 8)

rstyle(ax)

# plt.show()
####################################################################

## WAR BY POSITION #######################################

def by_position(hitters_zscore, stat, adj):
	pos = ('C','1B','2B','3B','SS','LF','CF','RF')
	new_pos = []
	for p in pos:
		new_pos.append('x'+p)

	pos_dict = {}
	for p in range(len(pos)):
		df = hitters_zscore[hitters_zscore[pos[p]] == 1]
		df['TOT'] = df[stat+'_Z']+df[stat+'_O']+df[stat+'_S']+df[stat+'_F']
		df['RANK'] = df.TOT.rank(ascending = False)
		df = df[df.RANK < 26]
		pos_dict[new_pos[p]] = np.mean([df[stat+'_Z'],df[stat+'_O'],df[stat+'_S'],df[stat+'_F']])

	ser = pd.Series(pos_dict)
	ser = ser.reindex(new_pos)
	return ser



pos_WAR = by_position(hitters_zscore, 'WAR', False)
pos_Fld = by_position(hitters_zscore, 'Fld', False)
pos_BsR = by_position(hitters_zscore, 'BsR', False)
pos_HR = by_position(hitters_zscore, 'HR', False)


pos = []
for p in pos_WAR.index:
	pos.append(p[1:])

ind = np.arange(1,9)

width = .20

fig = plt.figure(figsize = (12,9))
ax = fig.add_subplot(111)

set2 = brewer2mpl.get_map('PRGn', 'diverging', 4).mpl_colors

barh(ind,pos_WAR,width, color = set2[0], label = 'WAR')
barh(ind+width,pos_Fld,width, color = set2[1], label = 'Fielding')
barh(ind+width*2,pos_BsR,width, color = set2[2], label = 'Baserunning')
barh(ind+width*3,pos_HR,width, color = set2[3], label = 'HR')

suptitle('Players Grouped By Position \n Top 25 Players For Various Metrics', fontsize=14, fontweight='bold')
ax.legend(loc = 'best', fontsize = 14)
xlabel('Z-Scores', fontweight = 'bold')
ylim([1-width,9.25])
yticks(ind+2*width,pos, rotation = 0, fontsize = 14, fontweight = 'bold')
ax.invert_yaxis()

for label in ax.get_xticklines():
    label.set_visible(False)

for label in ax.get_yticklines():
    label.set_visible(False)

rstyle(ax)

# plt.show()


## TOP 50 BY HR (unlabeled) #######################################
top50 = hitters[['WAR_Z','WAR_O','WAR_F','WAR_S','Name']].copy().dropna()

top50['SUM'] = (top50.WAR_Z + top50.WAR_O + top50.WAR_F + top50.WAR_S)
top50['RANK'] = top50.SUM.rank(ascending = False)
top50 = top50.sort('SUM', ascending = False)
top50['RANK'] = range(1,len(top50)+1)
top50 = top50[top50.RANK<51]

last = lambda x: x.split(' ')[1]
last_name = top50.Name.apply(last)


ind = np.arange(1,51)

width = .7

fig = plt.figure()
ax = fig.add_subplot(111)

set2 = brewer2mpl.get_map('YlGnBu', 'sequential', 4).mpl_colors

bar(ind,top50.WAR_O/4,width, color = set2[3], label = 'Oliver')
bar(ind,top50.WAR_Z/4,width, color = set2[2], label = 'ZiPS', bottom = top50.WAR_O/4)
bar(ind,top50.WAR_S/4,width, color = set2[1], label = 'Steamer', bottom = (top50.WAR_O+top50.WAR_Z)/4)
bar(ind,top50.WAR_F/4,width, color = set2[0], label = 'Fans', bottom = (top50.WAR_O+top50.WAR_Z+top50.WAR_S)/4)

suptitle('Top 50 Players By Average WAR', fontsize=14, fontweight='bold')
ax.legend(loc = 'best', fontsize = 14)
ylabel('Average WAR')
xlim([1,50.75])
ax.tick_params(axis = 'x', which = 'both', bottom = 'off')

rstyle(ax)

# plt.show()
####################################################################

## TOP 50 BY HR ####################################################
top50 = hitters[['HR_ADJ_Z','HR_ADJ_O','HR_ADJ_F','HR_ADJ_S','Name']].copy().dropna()
top50['SUM'] = (top50.HR_ADJ_Z + top50.HR_ADJ_O + top50.HR_ADJ_F + top50.HR_ADJ_S)
top50['RANK'] = top50.SUM.rank(ascending = False)
top50 = top50.sort('SUM', ascending = False)
top50['RANK'] = range(1,len(top50)+1)
top50 = top50[top50.RANK<51]

ind = np.arange(1,51)

width = 1

fig = plt.figure(figsize = (12,9))
ax = fig.add_subplot(111)

set2 = brewer2mpl.get_map('YlGnBu', 'sequential', 4).mpl_colors

barh(ind,top50.HR_ADJ_O/4,width, color = set2[3], label = 'Oliver')
barh(ind,top50.HR_ADJ_Z/4,width, color = set2[2], label = 'ZiPS', left = top50.HR_ADJ_O/4)
barh(ind,top50.HR_ADJ_S/4,width, color = set2[1], label = 'Steamer', left = (top50.HR_ADJ_O+top50.HR_ADJ_Z)/4)
barh(ind,top50.HR_ADJ_F/4,width, color = set2[0], label = 'Fans', left = (top50.HR_ADJ_O+top50.HR_ADJ_Z+top50.HR_ADJ_S)/4)

suptitle('Top 50 Players By Average HR', fontsize=14, fontweight='bold')
ax.legend(loc = 'best', fontsize = 14)
xlabel('Average HR',fontweight='bold')
ylim([1,52])
yticks(ind+width/1.5,top50.Name, rotation = 0, fontsize = 8)
ax.invert_yaxis()

rstyle(ax)
# plt.show()
####################################################################
## TOP 10 BY WAR (labeled) #########################################

def top10(hitters, stat, adj):
	top10 = hitters.copy()

	if adj == True:
		top10['SUM'] = top10[stat + '_ADJ_F'] + top10[stat + '_ADJ_O']+ top10[stat + '_ADJ_S']+ top10[stat + '_ADJ_Z'] 
	else:
		top10['SUM'] = top10[stat + '_F'] + top10[stat + '_O']+ top10[stat + '_S']+ top10[stat + '_Z'] 

	top10['RANK'] = top10.SUM.rank(ascending = False)

	top10 = top10[top10.RANK<11].sort('RANK')

	last = lambda x: (x.split(' ')[0][:1] + '. ' + x.split(' ')[1]) if len(x.split(' ')) <3 else (x.split(' ')[0][:1] + '. ' + x.split(' ')[2]) 
	last_name = top10.Name.apply(last)

	ind = np.arange(1,11)

	width = .7

	fig = plt.figure()
	ax = fig.add_subplot(111)

	set2 = brewer2mpl.get_map('YlGnBu', 'sequential', 4).mpl_colors

	if adj == True:
		bar(ind,top10[stat + '_ADJ_O'],width, color = set2[3], label = 'Oliver')
		bar(ind,top10[stat + '_ADJ_Z'],width, color = set2[2], label = 'ZiPS', bottom = top10[stat + '_ADJ_O'])
		bar(ind,top10[stat + '_ADJ_S'],width, color = set2[1], label = 'Steamer', bottom = top10[stat + '_ADJ_O']+top10[stat + '_ADJ_Z'])
		bar(ind,top10[stat + '_ADJ_F'],width, color = set2[0], label = 'Fans', bottom = top10[stat + '_ADJ_O']+top10[stat + '_ADJ_Z']+top10[stat + '_ADJ_S'])
	else:
		bar(ind,top10[stat + '_O'],width, color = set2[3], label = 'Oliver')
		bar(ind,top10[stat + '_Z'],width, color = set2[2], label = 'ZiPS', bottom = top10[stat + '_O'])
		bar(ind,top10[stat + '_S'],width, color = set2[1], label = 'Steamer', bottom = top10[stat + '_O']+top10[stat + '_Z'])
		bar(ind,top10[stat + '_F'],width, color = set2[0], label = 'Fans', bottom = top10[stat + '_O']+top10[stat + '_Z']+top10[stat + '_S'])


	suptitle('Top 10 Players By ' + stat, fontsize=14, fontweight='bold')
	ax.legend(loc = 'best', fontsize = 14)
	ylabel('Total ' + stat + ' (Z-Scores)', fontweight = 'bold')
	xlim([1,10.75])
	ax.tick_params(axis = 'x', which = 'both', bottom = 'off')
	xticks(ind+width/2,last_name, rotation = 30, fontsize = 8)

	for label in ax.get_yticklines():
		label.set_visible(False)

	rstyle(ax)

	# plt.show()

top10(hitters_zscore, 'WAR', False)
top10(hitters_zscore, 'Fld', False)
top10(hitters_zscore, 'BsR', False)
top10(hitters_zscore, 'HR', True)
top10(hitters_zscore, 'SB', True)
top10(hitters_zscore, 'wOBA', False)
top10(hitters_zscore, 'OBP', False)
top10(hitters_zscore, 'SLG', False)
top10(hitters_zscore, 'SO', True)

plt.show()