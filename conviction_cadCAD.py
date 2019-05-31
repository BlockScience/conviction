#!/usr/bin/env python
# coding: utf-8

# In[1]:


from cadCAD.engine import ExecutionMode, ExecutionContext, Executor
from cadCAD.configuration import Configuration
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts
import pandas as pd
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.animation as animation
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.stats import expon, gamma


# This notebook uses the differential games framework developed by BlockScience. It is currently in private beta, and building towards a full open source release.
# 
# **Description:**
# 
# cadCAD is a Python library that assists in the processes of designing, testing and validating complex systems through simulation. At its core, cadCAD is a differential games engine that supports parameter sweeping and Monte Carlo analyses and can be easily integrated with other scientific computing Python modules and data science workflows.
# 
# To learn more about cadCAD, follow our [tutorial series](https://github.com/BlockScience/cadCAD-Tutorials/tree/master/01%20Tutorials)
# 
# **Installing cadCAD:**
# 
# cadCAD is in private beta. Tokens are issued to participants. Replace `<TOKEN>` in the installation URL below
# ```bash
# pip3 install cadCAD --extra-index-url https://<TOKEN>@repo.fury.io/blockscience/
# ```
# 
# If you'd like to participate in the beta program, contact cadcad [at] block [dot] science.
# 

# In[2]:


#helper functions
def get_nodes_by_type(g, node_type_selection):
    return [node for node in g.nodes if g.nodes[node]['type']== node_type_selection ]

def get_edges_by_type(g, edge_type_selection):
    return [edge for edge in g.edges if g.edges[edge]['type']== edge_type_selection ]


# In[3]:


#THIS policy is one of the main paramters of this system!

#maximum share of funds a proposal can take
beta = .2 #later we should set this to be param so we can sweep it
# tuning param for the trigger function
rho = .001

def trigger_threshold(requested, funds, supply):
    
    share = requested/funds
    if share < beta:
        return rho*supply/(beta-share)**2
    else: 
        return np.inf


# Note from Kris, consider: substitutibility of proposals st when a substitute passes, affinity for the others goes away; this will make the process more realistic because proposals will end up never passing.
# 
# implementation notes:
# - create substitutability matrix (proposal x proposal)
# - update accounting when thing pass: change affinities and should affect sentiments
# - define a new 'type' of proposals for tracking 'dead' ones (no longer candidates = zero staked)
# 

# In[4]:


#generate an initial set of 'n' participants
network = nx.DiGraph()
n = 100
for i in range(n):
    network.add_node(i)
    network.nodes[i]['type']="participant"
    
    h_rv = expon.rvs(loc=0.0, scale=1000)
    network.nodes[i]['holdings'] = h_rv
    
    s_rv = np.random.rand() 
    network.nodes[i]['sentiment'] = s_rv

participants = get_nodes_by_type(network, 'participant')
initial_supply = np.sum([ network.nodes[i]['holdings'] for i in participants])
print(initial_supply)

initial_funds = initial_supply


# In[5]:


#generate initial proposals
m = 7
for ind in range(m):
    j = n+ind
    network.add_node(j)
    network.nodes[j]['type']="proposal"
    network.nodes[j]['conviction']=0
    network.nodes[j]['status']='candidate'
    network.nodes[j]['age']=0
    
    r_rv = gamma.rvs(3,loc=0.001, scale=10000)
    network.node[j]['funds_requested'] = r_rv
    
    network.nodes[j]['trigger']= trigger_threshold(r_rv, initial_funds, initial_supply)
    
    for i in range(n):
        network.add_edge(i, j)
        
        rv = np.random.rand()
        a_rv = 1-4*(1-rv)*rv #polarized distribution
        network.edges[(i, j)]['affinity'] = a_rv
        network.edges[(i,j)]['tokens'] = 0
        network.edges[(i, j)]['conviction'] = 0

proposals = get_nodes_by_type(network, 'proposal')
total_requested = np.sum([ network.nodes[i]['funds_requested'] for i in proposals])
print(total_requested/initial_funds)


# In[6]:


network.nodes[get_nodes_by_type(network, 'proposal')[0]]


# In[7]:


network.nodes[get_nodes_by_type(network, 'participant')[0]]


# In[8]:


plt.hist([ network.nodes[i]['holdings'] for i in participants])
plt.title('Histogram of Participants Token Holdings')


# In[9]:


plt.hist([ network.nodes[i]['funds_requested'] for i in proposals])
plt.title('Histogram of Proposals Funds Requested')


# In[10]:


plt.hist([ network.edges[e]['affinity'] for e in network.edges])
plt.title('Histogram of Affinities between Participants and Proposals')


# In[11]:


plt.hist([ network.edges[e]['affinity'] for e in network.edges], weights = [network.nodes[e[0]]['holdings']for e in network.edges],alpha = 1)
plt.title('Histogram of Affinities between Participants and Proposals weighted by holdings')


# In[12]:


T = 200 #iterations of graph update in our simulation
#param for conviction accumilation
alpha = .5 #later we should set this to be param so we can sweep it

#sentiment of the outside world which drives grants
initial_sentiment = .8

#sentiment decay rate
mu =.001 #later we should set this to be param so we can sweep it

#minimum periods passed before a proposal can pass
tmin = 7

#how close to a participant's highest affinity proposal that
#another proposal has to be for them to be happy about its advancement
sensitivity = .75

min_completion_rate = 10


# In[13]:


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Settings of general simulation parameters, unrelated to the system itself
# `T` is a range with the number of discrete units of time the simulation will run for;
# `N` is the number of times the simulation will be run (Monte Carlo runs)
# We'll cover the `M` key in a future article. For now, let's leave it empty
simulation_parameters = {
    'T': range(T),
    'N': 1,
    'M': {}
}


# In[14]:


initial_conditions = {'network':network,
                      'supply': initial_supply,
                      'funds':initial_funds,
                      'sentiment': initial_sentiment}


# In[15]:


#functions for partial state update block 1

def gen_new_participant(network, new_participant_holdings):
    
    i = len([node for node in network.nodes])
    
    network.add_node(i)
    network.nodes[i]['type']="participant"
    
    s_rv = np.random.rand() 
    network.nodes[i]['sentiment'] = s_rv
    network.nodes[i]['holdings']=new_participant_holdings
    
    for j in get_nodes_by_type(network, 'proposal'):
        network.add_edge(i, j)
        
        rv = np.random.rand()
        a_rv = 1-4*(1-rv)*rv #polarized distribution
        network.edges[(i, j)]['affinity'] = a_rv
        network.edges[(i,j)]['tokens'] = a_rv*network.nodes[i]['holdings']
        network.edges[(i, j)]['conviction'] = 0
    
    return network
    

def gen_new_proposal(network, funds, supply):
    j = len([node for node in network.nodes])
    network.add_node(j)
    network.nodes[j]['type']="proposal"
    
    network.nodes[j]['conviction']=0
    network.nodes[j]['status']='candidate'
    network.nodes[j]['age']=0
    
    rescale = 10000*funds/initial_funds
    r_rv = gamma.rvs(3,loc=0.001, scale=rescale)
    network.node[j]['funds_requested'] = r_rv
    
    network.nodes[j]['trigger']= trigger_threshold(r_rv, funds, supply)
    
    participants = get_nodes_by_type(network, 'participant')
    proposing_participant = np.random.choice(participants)
    
    for i in participants:
        network.add_edge(i, j)
        if i==proposing_participant:
            network.edges[(i, j)]['affinity']=1
        else:
            rv = np.random.rand()
            a_rv = 1-4*(1-rv)*rv #polarized distribution
            network.edges[(i, j)]['affinity'] = a_rv
            
        network.edges[(i, j)]['conviction'] = 0
        network.edges[(i,j)]['tokens'] = 0
    return network
        
        

def driving_process(params, step, sL, s):
    
    #placeholder plumbing for random processes
    arrival_rate = 10/s['sentiment']
    rv1 = np.random.rand()
    new_participant = bool(rv1<1/arrival_rate)
    if new_participant:
        h_rv = expon.rvs(loc=0.0, scale=1000)
        new_participant_holdings = h_rv
    else:
        new_participant_holdings = 0
    
    network = s['network']
    affinities = [network.edges[e]['affinity'] for e in network.edges ]
    median_affinity = np.median(affinities)
    
    proposals = get_nodes_by_type(network, 'proposal')
    fund_requests = [network.nodes[j]['funds_requested'] for j in proposals if network.nodes[j]['status']=='candidate' ]
    
    funds = s['funds']
    total_funds_requested = np.sum(fund_requests)
    
    proposal_rate = 10/median_affinity * total_funds_requested/funds
    rv2 = np.random.rand()
    new_proposal = bool(rv2<1/proposal_rate)
    
    sentiment = s['sentiment']
    funds = s['funds']
    scale_factor = 1+4000*sentiment**2
    
    #this shouldn't happen but expon is throwing domain errors
    if scale_factor > 1: 
        funds_arrival = expon.rvs(loc = 0, scale = scale_factor )
    else:
        funds_arrival = 0
    
    return({'new_participant':new_participant,
            'new_participant_holdings':new_participant_holdings,
            'new_proposal':new_proposal, 
            'funds_arrival':funds_arrival})

def update_network(params, step, sL, s, _input):
    
    network = s['network']
    funds = s['funds']
    supply = s['supply']
    #placeholder plumbing for new proposals and new participants
    new_participant = _input['new_participant'] #T/F
    new_proposal = _input['new_proposal'] #T/F
    # IF THEN logic to create new nodes // left out for now since always FALSE
    if new_participant:
        new_participant_holdings = _input['new_participant_holdings']
        network = gen_new_participant(network, new_participant_holdings)
    
    if new_proposal:
        network= gen_new_proposal(network,funds,supply )
    
    #update age of the existing proposals
    proposals = get_nodes_by_type(network, 'proposal')
    
    for j in proposals:
        network.nodes[j]['age'] =  network.nodes[j]['age']+1
        if network.nodes[j]['status'] == 'candidate':
            requested = network.nodes[j]['funds_requested']
            network.nodes[j]['trigger'] = trigger_threshold(requested, funds, supply)
        else:
            network.nodes[j]['trigger'] = np.nan
            
    key = 'network'
    value = network
    
    return (key, value)

def increment_funds(params, step, sL, s, _input):
    
    funds = s['funds']
    funds_arrival = _input['funds_arrival']

    #increment funds
    funds = funds + funds_arrival
    
    key = 'funds'
    value = funds
    
    return (key, value)

def increment_supply(params, step, sL, s, _input):
    
    supply = s['supply']
    supply_arrival = _input['new_participant_holdings']

    #increment funds
    supply = supply + supply_arrival
    
    key = 'supply'
    value = supply
    
    return (key, value)


# In[16]:


#partial state update block 2
def check_progress(params, step, sL, s):
    
    network = s['network']
    proposals = get_nodes_by_type(network, 'proposal')
    
    completed = []
    for j in proposals:
        if network.nodes[j]['status'] == 'active':
            grant_size = network.nodes[j]['funds_requested']
            likelihood = 1.0/(min_completion_rate+np.log(grant_size))
            if np.random.rand() < likelihood:
                completed.append(j)
    
    return({'completed':completed})

def complete_proposal(params, step, sL, s, _input):
    
    network = s['network']
    participants = get_nodes_by_type(network, 'participant')
    
    completed = _input['completed']
    for j in completed:
        network.nodes[j]['status']='completed'
        for i in participants:
            force = network.edges[(i,j)]['affinity']
            sentiment = network.node[i]['sentiment']
            network.node[i]['sentiment'] = get_sentimental(sentiment, force, decay=False)
    
    key = 'network'
    value = network
    
    return (key, value)

def update_sentiment_on_completion(params, step, sL, s, _input):
    
    network = s['network']
    proposals = get_nodes_by_type(network, 'proposal')
    completed = _input['completed']
    
    grants_outstanding = np.sum([network.nodes[j]['funds_requested'] for j in proposals if network.nodes[j]['status']=='active'])
    
    grants_completed = np.sum([network.nodes[j]['funds_requested'] for j in completed])
    
    sentiment = s['sentiment']
    
    force = grants_completed/grants_outstanding
    if (force >=0) and (force <=1):
        sentiment = get_sentimental(sentiment, force, True)
    else:
        sentiment = get_sentimental(sentiment, 0, True)
    
    
    key = 'sentiment'
    value = sentiment
    
    return (key, value)

def get_sentimental(sentiment, force, decay=True):
    sentiment = sentiment*(1-int(decay)*mu) + force
    
    if sentiment > 1:
        sentiment = 1
        
    return sentiment


# In[17]:


#partial state update block 3
def trigger_function(params, step, sL, s):
    
    network = s['network']
    funds = s['funds']
    supply = s['supply']
    proposals = get_nodes_by_type(network, 'proposal')
    
    accepted = []
    triggers = {}
    for j in proposals:
        if network.nodes[j]['status'] == 'candidate':
            requested = network.nodes[j]['funds_requested']
            age = network.nodes[j]['age']
            threshold = trigger_threshold(requested, funds, supply)
            if age > tmin:
                conviction = network.nodes[j]['conviction']
                if conviction >threshold:
                    accepted.append(j)
        else:
            threshold = np.nan
            
        triggers[j] = threshold
                
        
                    
    return({'accepted':accepted, 'triggers':triggers})

def decrement_funds(params, step, sL, s, _input):
    
    funds = s['funds']
    network = s['network']
    accepted = _input['accepted']

    #decrement funds
    for j in accepted:
        funds = funds - network.nodes[j]['funds_requested']
    
    key = 'funds'
    value = funds
    
    return (key, value)

def update_proposals(params, step, sL, s, _input):
    
    network = s['network']
    accepted = _input['accepted']
    triggers = _input['triggers']
    participants = get_nodes_by_type(network, 'participant')
    proposals = get_nodes_by_type(network, 'proposals')
    
    for j in proposals:
        network.nodes[j]['trigger'] = triggers[j]
    
    #bookkeeping conviction and participant sentiment
    for j in accepted:
        network.nodes[j]['status']='active'
        network.nodes[j]['conviction']=np.nan
        #change status to active
        for i in participants:
        
            edge = (i,j)
            #reset tokens assigned to other candidates
            network.edges[(i,j)]['tokens']=0
            network.edges[(i,j)]['conviction'] = np.nan
            
            #update participants sentiments (positive or negative) 
            affinities = [network.edges[(i,p)]['affinity'] for p in proposals if not(p in accepted)]
            if len(affinities)>1:
                max_affinity = np.max(affinities)
                force = network.edges[(i,j)]['affinity']-sensitivity*max_affinity
            else:
                force = 0
            
            #based on what their affinities to the accepted proposals
            network.nodes[i]['sentiment'] = get_sentimental(network.nodes[i]['sentiment'], force, False)
            
    
    key = 'network'
    value = network
    
    return (key, value)

def update_sentiment_on_release(params, step, sL, s, _input):
    
    network = s['network']
    proposals = get_nodes_by_type(network, 'proposal')
    accepted = _input['accepted']
    
    proposals_outstanding = np.sum([network.nodes[j]['funds_requested'] for j in proposals if network.nodes[j]['status']=='candidate'])
    
    proposals_accepted = np.sum([network.nodes[j]['funds_requested'] for j in accepted])
    
    sentiment = s['sentiment']
    force = proposals_accepted/proposals_outstanding
    if (force >=0) and (force <=1):
        sentiment = get_sentimental(sentiment, force, False)
    else:
        sentiment = get_sentimental(sentiment, 0, False)
    
    key = 'sentiment'
    value = sentiment
    
    return (key, value)


# In[22]:


def participants_decisions(params, step, sL, s):
    
    network = s['network']
    participants = get_nodes_by_type(network, 'participant')
    proposals = get_nodes_by_type(network, 'proposal')
    candidates = [j for j in proposals if network.nodes[j]['status']=='candidate']
    
    gain = .01
    delta_holdings={}
    proposals_supported ={}
    for i in participants:
        force = network.nodes[i]['sentiment']-sensitivity
        delta_holdings[i] = network.nodes[i]['holdings']*gain*force
        
        support = []
        for j in candidates:
            affinity = network.edges[(i, j)]['affinity']
            cutoff = sensitivity*np.max([network.edges[(i,p)]['affinity'] for p in candidates])
            if cutoff <.5:
                cutoff = .5
            
            if affinity > cutoff:
                support.append(j)
        
        proposals_supported[i] = support
    
    return({'delta_holdings':delta_holdings, 'proposals_supported':proposals_supported})

def update_tokens(params, step, sL, s, _input):
    
    network = s['network']
    delta_holdings = _input['delta_holdings']
    proposals = get_nodes_by_type(network, 'proposal')
    proposals_supported = _input['proposals_supported']
    participants = get_nodes_by_type(network, 'participant')
    
    for i in participants:
        network.nodes[i]['holdings'] = network.nodes[i]['holdings']+delta_holdings[i]
        supported = proposals_supported[i]
        total_affinity = np.sum([ network.edges[(i, j)]['affinity'] for j in supported])
        for j in proposals:
            if j in supported:
                normalized_affinity = network.edges[(i, j)]['affinity']/total_affinity
                network.edges[(i, j)]['tokens'] = normalized_affinity*network.nodes[i]['holdings']
            else:
                network.edges[(i, j)]['tokens'] = 0
            
            prior_conviction = network.edges[(i, j)]['conviction']
            current_tokens = network.edges[(i, j)]['tokens']
            network.edges[(i, j)]['conviction'] =current_tokens+alpha*prior_conviction
    
    for j in proposals:
        network.nodes[j]['conviction'] = np.sum([ network.edges[(i, j)]['conviction'] for i in participants])
    
    key = 'network'
    value = network
    
    return (key, value)

def update_supply(params, step, sL, s, _input):
    
    supply = s['supply']
    delta_holdings = _input['delta_holdings']
    delta_supply = np.sum([v for v in delta_holdings.values()])
    
    supply = supply + delta_supply
    
    key = 'supply'
    value = supply
    
    return (key, value)


# In[23]:


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# The Partial State Update Blocks
partial_state_update_blocks = [
    { 
        'policies': { 
            #new proposals or new participants
            'random': driving_process
        },
        'variables': {
            'network': update_network,
            'funds':increment_funds,
            'supply':increment_supply
        }
    },
    {
      'policies': {
          'completion': check_progress #see if any of the funded proposals completes
        },
        'variables': { # The following state variables will be updated simultaneously
            'sentiment': update_sentiment_on_completion, #note completing decays sentiment, completing bumps it
            'network': complete_proposal #book-keeping
        }
    },
        {
      'policies': {
          'release': trigger_function #check each proposal to see if it passes
        },
        'variables': { # The following state variables will be updated simultaneously
            'funds': decrement_funds, #funds expended
            'sentiment': update_sentiment_on_release, #releasing funds can bump sentiment
            'network': update_proposals #reset convictions, and participants sentiments
                                        #update based on affinities
        }
    },
    { 
        'policies': { 
            'participants_act': participants_decisions, #high sentiment, high affinity =>buy
                                                        #low sentiment, low affinities => burn
                                                        #assign tokens to top affinities
        },
        'variables': {
            'supply': update_supply,
            'network': update_tokens #update everyones holdings 
                                    #and their conviction for each proposal
        }
    }
]


# In[24]:


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# The configurations above are then packaged into a `Configuration` object
config = Configuration(initial_state=initial_conditions, #dict containing variable names and initial values
                       partial_state_update_blocks=partial_state_update_blocks, #dict containing state update functions
                       sim_config=simulation_parameters #dict containing simulation parameters
                      )


# In[25]:


exec_mode = ExecutionMode()
exec_context = ExecutionContext(exec_mode.single_proc)
executor = Executor(exec_context, [config]) # Pass the configuration object inside an array
raw_result, tensor = executor.main() # The `main()` method returns a tuple; its first elements contains the raw results


# In[26]:


df = pd.DataFrame(raw_result)


# In[172]:


df.tail(5)


# In[28]:


df.supply.plot()


# In[29]:


df.sentiment.plot()


# In[30]:


df.plot(x='timestep', y='funds')


# In[31]:


def pad(vec, length,fill=True):
    
    if fill:
        padded = np.zeros(length,)
    else:
        padded = np.empty(length,)
        padded[:] = np.nan
        
    for i in range(len(vec)):
        padded[i]= vec[i]
        
    return padded

def make2D(key, data, fill=False):
    maxL = data[key].apply(len).max()
    newkey = 'padded_'+key
    data[newkey] = data[key].apply(lambda x: pad(x,maxL,fill))
    reshaped = np.array([a for a in data[newkey].values])
    
    return reshaped


# In[32]:


df['conviction'] = df.network.apply(lambda g: np.array([g.nodes[j]['conviction'] for j in get_nodes_by_type(g, 'proposal') if g.nodes[j]['status']=='candidate']))
df['candidate_count'] = df.network.apply(lambda g: len([j for j in get_nodes_by_type(g, 'proposal') if g.nodes[j]['status']=='candidate']))
df['candidate_funds'] = df.network.apply(lambda g: np.sum([g.nodes[j]['funds_requested'] for j in get_nodes_by_type(g, 'proposal') if g.nodes[j]['status']=='candidate']))
df['candidate_funds_requested'] = df.network.apply(lambda g: np.array([g.nodes[j]['funds_requested'] for j in get_nodes_by_type(g, 'proposal') if g.nodes[j]['status']=='candidate']))
df['active_count'] = df.network.apply(lambda g: len([j for j in get_nodes_by_type(g, 'proposal') if g.nodes[j]['status']=='active']))
df['active_funds'] = df.network.apply(lambda g: np.sum([g.nodes[j]['funds_requested'] for j in get_nodes_by_type(g, 'proposal') if g.nodes[j]['status']=='active']))
df['completed_count'] = df.network.apply(lambda g: len([j for j in get_nodes_by_type(g, 'proposal') if g.nodes[j]['status']=='completed']))
df['completed_funds'] = df.network.apply(lambda g: np.sum([g.nodes[j]['funds_requested'] for j in get_nodes_by_type(g, 'proposal') if g.nodes[j]['status']=='completed']))


# In[33]:


df['funds_requested'] = df.network.apply(lambda g: np.array([g.nodes[j]['funds_requested'] for j in get_nodes_by_type(g, 'proposal')]))
df['share_of_funds_requested'] = df.candidate_funds_requested/df.funds

df['share_of_funds_requested_all'] = df.funds_requested/df.funds


# In[34]:


df['triggers'] = df.network.apply(lambda g: np.array([g.nodes[j]['trigger'] for j in get_nodes_by_type(g, 'proposal') if g.nodes[j]['status']=='candidate' ]))
df['conviction_share_of_trigger'] = df.conviction/df.triggers
df['age'] = df.network.apply(lambda g: np.array([g.nodes[j]['age'] for j in get_nodes_by_type(g, 'proposal') if g.nodes[j]['status']=='candidate' ]))


# In[35]:


df['age_all'] = df.network.apply(lambda g: np.array([g.nodes[j]['age'] for j in get_nodes_by_type(g, 'proposal') ]))
df['conviction_all'] = df.network.apply(lambda g: np.array([g.nodes[j]['conviction'] for j in get_nodes_by_type(g, 'proposal') ]))
df['triggers_all'] = df.network.apply(lambda g: np.array([g.nodes[j]['trigger'] for j in get_nodes_by_type(g, 'proposal')  ]))

df['conviction_share_of_trigger_all'] = df.conviction_all/df.triggers_all


# In[36]:


rdf= df[df.substep==4].copy()


# In[37]:


last_net= df.network.values[-1]
last_props=get_nodes_by_type(last_net, 'proposal')
M = len(last_props)
last_parts=get_nodes_by_type(last_net, 'participant')
N = len(last_parts)


# In[38]:


affinities = np.empty((N,M))


# In[39]:


for i_ind in range(N):
    for j_ind in range(M):
        i = last_parts[i_ind]
        j = last_props[j_ind]
        affinities[i_ind][j_ind] = last_net.edges[(i,j)]['affinity']


# In[40]:


dims = (100, 25)
fig, ax = plt.subplots(figsize=dims)

sns.heatmap(affinities,
            yticklabels=last_parts,
            xticklabels=last_props,
            square=True,
            cbar=True,
            ax=ax)

plt.title('affinities between participants and proposals')
plt.xlabel('proposal_id')
plt.ylabel('participant_id')


# In[41]:


#working on deduplicating colors
#
#last_props=get_nodes_by_type(last_net, 'proposal')
#M = len(last_props)

#cm = plt.get_cmap('gist_rainbow')
#c= [cm(1.*j/M) for j in range(M)] 


# In[168]:


make2D('age_all', rdf)


# In[170]:


plt.plot(rdf.timestep,make2D('age_all', rdf))
plt.title('check age')


# In[43]:


rdf.plot(x='timestep',y=['candidate_count','active_count','completed_count'])


# In[44]:


rdf.plot(x='timestep',y=['candidate_funds','active_funds','completed_funds'])


# In[45]:


plt.semilogy(rdf.timestep,make2D('conviction_all', rdf))
plt.title('conviction by proposal')
plt.xlabel('time $t$')
plt.ylabel('conviction')


# In[46]:


plt.semilogy(make2D('age_all', rdf),make2D('conviction_all', rdf))
plt.title('conviction by proposal')
plt.xlabel('proposal age')
plt.ylabel('conviction')


# In[47]:


plt.plot(rdf.timestep,make2D('share_of_funds_requested_all', rdf))
plt.title('share_of_funds_requested by proposal')
plt.xlabel('time $t$')
plt.ylabel('share_of_funds_requested')


# In[48]:


plt.semilogy(make2D('age_all', rdf),make2D('share_of_funds_requested_all', rdf))
plt.title('share_of_funds_requested by proposal')
plt.xlabel('proposal age')
plt.ylabel('share_of_funds_requested')


# In[166]:


plt.loglog(make2D('share_of_funds_requested_all', rdf), make2D('conviction_all', rdf), '.')
plt.ylabel('conviction')
plt.xlabel('share_of_funds_requested')


# In[50]:


plt.semilogy(make2D('age_all', rdf), make2D('triggers_all', rdf))
plt.ylabel('triggers')
plt.xlabel('proposal_age')


# In[51]:


plt.loglog(make2D('conviction_all', rdf), make2D('triggers_all', rdf))
a = plt.axis()
plt.loglog(a[:2],a[2:], 'k',alpha=.5 )
plt.ylabel('triggers')
plt.xlabel('conviction')
plt.title('phase: Triggers & Conviction')


# In[174]:


plt.semilogy(rdf.timestep,make2D('conviction_share_of_trigger_all', rdf))
plt.title('conviction_share_of_trigger')
plt.xlabel('time $t$')
plt.ylabel('conviction_share_of_trigger')
plt.hlines(1,0,T, linestyle='--')


# In[63]:


plt.semilogy(make2D('age_all', rdf), make2D('conviction_share_of_trigger_all', rdf))
plt.ylabel('triggers')
plt.xlabel('proposal_age')
plt.hlines(1,0,T, linestyle='--')


# In[54]:


pos = {}
for ind in range(N):
    i = last_parts[ind] 
    pos[i] = np.array([0, 2*ind-N])

for ind in range(M):
    j = last_props[ind] 
    pos[j] = np.array([1, 2*N/M *ind-N])

#for i in last_parts:
#for j in last_props:
        


# In[55]:


edges = [e for e in last_net.edges]
max_tok = np.max([last_net.edges[e]['tokens'] for e in edges])

E = len(edges)

node_color = np.empty((M+N,4))
node_size = np.empty(M+N)

edge_color = np.empty((E,4))
cm = plt.get_cmap('Reds')

cNorm  = colors.Normalize(vmin=0, vmax=max_tok)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)


# In[107]:


size_scale = 1/500
node_label = {}

for j in last_props:
    node_size[j] = last_net.nodes[j]['funds_requested']*size_scale
    if last_net.nodes[j]['status']=="candidate":
        node_color[j] = colors.to_rgba('blue')
        trigger = last_net.nodes[j]['trigger']
        #print(trigger)
        conviction = last_net.nodes[j]['conviction']
        #print(conviction)
        percent_of_trigger = str(int(100*conviction/trigger))+'%'
        #age = last_net.nodes[j]['age']
        node_label[j] = str(percent_of_trigger)
    elif last_net.nodes[j]['status']=="active":
        node_color[j] = colors.to_rgba('orange')
        node_label[j] = ''
    elif last_net.nodes[j]['status']=="completed":
        node_color[j] = colors.to_rgba('green')
        node_label[j] = ''

for i in last_parts:    
    node_size[i] = last_net.nodes[i]['holdings']*size_scale
    node_color[i] = colors.to_rgba('red')
    node_label[i] = ''

included_edges = []
for ind in range(E):
    e = edges[ind]
    tokens = last_net.edges[e]['tokens']
    if tokens >0:
        included_edges.append(e)
    #print(tokens)
    edge_color[ind] = scalarMap.to_rgba(tokens)

iE = len(included_edges)
included_edge_color = np.empty((iE,4))
for ind in range(iE):
    e = included_edges[ind]
    tokens = last_net.edges[e]['tokens']
    included_edge_color[ind] = scalarMap.to_rgba(tokens)


# In[108]:


nx.draw(last_net,
        pos=pos, 
        node_size = node_size, 
        node_color = node_color, 
        edge_color = included_edge_color, 
        edgelist=included_edges,
        labels = node_label)
plt.title('Tokens Staked by Partipants to Proposals')


# In[125]:


nets = rdf.network.values


# In[164]:


def snap_plot(nets, size_scale = 1/500, ani = False, dims = (20,20), savefigs=False ):
    
    last_net= df.network.values[-1]
    last_props=get_nodes_by_type(last_net, 'proposal')
    M = len(last_props)
    last_parts=get_nodes_by_type(last_net, 'participant')
    N = len(last_parts)
    pos = {}
    
    for ind in range(N):
        i = last_parts[ind] 
        pos[i] = np.array([0, 2*ind-N])

    for ind in range(M):
        j = last_props[ind] 
        pos[j] = np.array([1, 2*N/M *ind-N])
    
    if ani:
        figs = []
        fig, ax = plt.subplots(figsize=dims)
    
    if savefigs:
        counter = 0
        length = 10
        import string
        unique_id = ''.join([np.random.choice(list(string.ascii_letters + string.digits)) for _ in range(length)])
    for net in nets:
        edges = [e for e in net.edges]
        max_tok = np.max([net.edges[e]['tokens'] for e in edges])

        E = len(edges)
        
        net_props = get_nodes_by_type(net, 'proposal')
        net_parts = get_nodes_by_type(net, 'participant')
        net_node_label ={}
        
        num_nodes = len([node for node in net.nodes])
        
        node_color = np.empty((num_nodes,4))
        node_size = np.empty(num_nodes)

        edge_color = np.empty((E,4))
        cm = plt.get_cmap('Reds')

        cNorm  = colors.Normalize(vmin=0, vmax=max_tok)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    
    

        for j in net_props:
            node_size[j] = net.nodes[j]['funds_requested']*size_scale
            if net.nodes[j]['status']=="candidate":
                node_color[j] = colors.to_rgba('blue')
                trigger = net.nodes[j]['trigger']      
                conviction = net.nodes[j]['conviction']
                percent_of_trigger = "          "+str(int(100*conviction/trigger))+'%'
                net_node_label[j] = str(percent_of_trigger)
            elif net.nodes[j]['status']=="active":
                node_color[j] = colors.to_rgba('orange')
                net_node_label[j] = ''
            elif net.nodes[j]['status']=="completed":
                node_color[j] = colors.to_rgba('green')
                net_node_label[j] = ''

        for i in net_parts:    
            node_size[i] = net.nodes[i]['holdings']*size_scale
            node_color[i] = colors.to_rgba('red')
            net_node_label[i] = ''

        included_edges = []
        for ind in range(E):
            e = edges[ind]
            tokens = net.edges[e]['tokens']
            if tokens >0:
                included_edges.append(e)
            edge_color[ind] = scalarMap.to_rgba(tokens)

        iE = len(included_edges)
        included_edge_color = np.empty((iE,4))
        for ind in range(iE):
            e = included_edges[ind]
            tokens = net.edges[e]['tokens']
            included_edge_color[ind] = scalarMap.to_rgba(tokens)
        
        nx.draw(net,
                pos=pos, 
                node_size = node_size,
                node_color = node_color, 
                edge_color = included_edge_color, 
                edgelist=included_edges,
                labels = net_node_label)
        plt.title('Tokens Staked by Partipants to Proposals')
        if ani:
            nx.draw(net,
                    pos=pos, 
                    node_size = node_size,
                    node_color = node_color, 
                    edge_color = included_edge_color, 
                    edgelist=included_edges,
                    labels = net_node_label, ax=ax)
            figs.append(fig)
            
        else:
            nx.draw(net,
                pos=pos, 
                node_size = node_size,
                node_color = node_color, 
                edge_color = included_edge_color, 
                edgelist=included_edges,
                labels = net_node_label)
            plt.title('Tokens Staked by Partipants to Proposals')
            if savefigs:
                plt.savefig(unique_id+'_fig'+str(counter)+'.png')
                counter = counter+1
            plt.show()
        
    if ani:
        False
        #anim = animation.ArtistAnimation(fig, , interval=50, blit=True, repeat_delay=1000)
        #plt.show()
            


# In[165]:


snap_plot(nets, ani=False, savefigs=False)


# In[143]:


#totally failing at animating by trying to save a sequence of figures.
#snap_plot(nets, ani=True)

#saving the images to files works so there is almost the option to compile a video from the images


# In[ ]:




