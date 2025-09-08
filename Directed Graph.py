import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import community as community_louvain


df = pd.read_excel('top200_with_follow_relationships(final).xlsx')
np.random.seed(42)

G = nx.DiGraph()


for _, row in df.iterrows():
    G.add_node(row['Username'], 
               followers=row['Followers'],
               posts=row['Posts'],
               engagement_rate=row['Engagement Rate'],
               country=row['Country'],
               main_topic=row['Main topic'])

for i, row in df.iterrows():
    current_user = row['Username']
    current_followers = row['Followers']
    
   # Find all users who have more followers than the current user
    potential_following = df[df['Followers'] > current_followers]
    
    # Randomly select 1-3 users to track
    if len(potential_following) > 0:
        num_to_follow = np.random.randint(1, min(4, len(potential_following)+1))
        following = np.random.choice(potential_following['Username'], 
                                    num_to_follow, 
                                    replace=False)
        
        # Adding directed edges
        for user in following:
            G.add_edge(current_user, user)

# Calculating network metrics
betweenness = nx.betweenness_centrality(G)
pagerank = nx.pagerank(G)
clustering = nx.clustering(G)


for node in G.nodes():
    G.nodes[node]['betweenness'] = betweenness.get(node, 0)
    G.nodes[node]['pagerank'] = pagerank.get(node, 0)
    G.nodes[node]['clustering'] = clustering.get(node, 0)








#Visualization method
#Draw a network diagram

#top 30
top_users = sorted(G.nodes(data=True), key=lambda x: x[1]['followers'], reverse=True)[:30]
top_usernames = [user[0] for user in top_users]
subgraph = nx.DiGraph()

for username in top_usernames:
    subgraph.add_node(username, **G.nodes[username])

#Add edges to these nodes (track relationships)
for username in top_usernames:
    # Add the relationship (outgoing edge) that the user is tracking
    for followed in G.successors(username):
        if followed in top_usernames:  
            subgraph.add_edge(username, followed)
    
    # incoming edge
    for follower in G.predecessors(username):
        if follower in top_usernames:
            subgraph.add_edge(follower, username)


fig, ax = plt.subplots(figsize=(10, 5))
node_size = [np.log(subgraph.nodes[node]['followers'])*30 for node in subgraph.nodes()]
# Node color based on interaction rate
engagement_rates = [subgraph.nodes[node]['engagement_rate'] for node in subgraph.nodes()]
norm = plt.Normalize(min(engagement_rates), max(engagement_rates))
cmap = plt.cm.plasma  

pos = nx.spring_layout(subgraph, k=0.5, iterations=100, seed=42)


nodes = nx.draw_networkx_nodes(subgraph, pos, node_size=node_size, 
                              node_color=engagement_rates, 
                              cmap=cmap, alpha=0.9, 
                              linewidths=1, edgecolors='gray', ax=ax)


edges = nx.draw_networkx_edges(subgraph, pos, edge_color='black', 
                              alpha=0.4, arrowstyle='-|>', 
                              arrowsize=15, width=1.5,
                              connectionstyle='arc3,rad=0.1', ax=ax)


label_map = {node: f"User{str(i+1)}" for i, node in enumerate(subgraph.nodes())}
nx.draw_networkx_labels(subgraph, pos, labels=label_map, font_size=10, 
                       font_family='sans-serif', font_weight='bold')


sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
cbar.set_label('Engagement Rate', fontsize=12)

ax.set_title('Instagram Top 30 Users and Their Following Network', size=18, pad=20)
plt.suptitle('Node size represents follower count | Color represents engagement rate | Arrows show following direction', 
             y=0.97, fontsize=12)

plt.text(0.05, 0.05, 
         "Community Structure:\n"
         "- Nodes represent users\n"
         "- Size ∝ log(followers)\n" 
         "- Color = engagement rate\n"
         "- Arrows: follower → followed", 
         transform=plt.gcf().transFigure,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
         fontsize=10)

ax.axis('off')
plt.tight_layout()
plt.show()


print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
for node in G.nodes():
    G.nodes[node]['clustering'] = clustering.get(node, 0)


print("\nClustering coefficient per user:")
for node, value in clustering.items():
    print(f"{node}: {value:.4f}")

print(f"Average clustering coefficient: {nx.average_clustering(G)}")
print(f"Density: {nx.density(G)}")




#Centrality analysis
#Calculate degree centrality
in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())

#Find the most influential users
top_in_degree = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:30]
top_out_degree = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:30]
top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:30]

print("\nTop 30 users by in-degree (most followed):")
for user, degree in top_in_degree:
    print(f"{user}: {degree}")

print("\nTop 30 users by out-degree (following most):")
for user, degree in top_out_degree:
    print(f"{user}: {degree}")

print("\nTop 30 users by PageRank:")
for user, score in top_pagerank:
    print(f"{user}: {score:.4f}")


#Community Detection(all)
import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx

#undirected
G_undirected = G.to_undirected()

#Louvain
partition = community_louvain.best_partition(G_undirected)


for node in G.nodes():
    G.nodes[node]['community'] = partition.get(node, -1)

num_communities = len(set(partition.values()))
print(f"\nNumber of communities detected: {num_communities}")


pos = nx.spring_layout(G_undirected, k=0.5, iterations=100, seed=42)


node_size = [np.log(G.nodes[node]['followers'] + 1)*30 for node in G.nodes()]

plt.figure(figsize=(15, 8))
cmap = plt.cm.tab20  

nx.draw_networkx_nodes(G, pos, node_size=node_size, 
                       node_color=[partition[node] for node in G.nodes()], 
                       cmap=cmap, alpha=0.8, edgecolors='black', linewidths=0.5)
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, arrows=False)
nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

plt.title('Instagram User Communities', size=15)
plt.axis('off')
plt.tight_layout()
plt.show()

followers_dict = nx.get_node_attributes(G, 'followers')
betweenness_dict = nx.get_node_attributes(G, 'betweenness')

top_betweenness = sorted(betweenness_dict.items(), key=lambda x: x[1], reverse=True)[:200]

print("\nTop 30 users by Betweenness Centrality with Followers:")
print(f"{'Username':<20} {'Followers':>10} {'Betweenness':>15}")
for user, bc in top_betweenness:
    print(f"{user:<20} {followers_dict.get(user, 0):>10} {bc:>15.6f}")



