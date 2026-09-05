#include <vector>
#include <queue>
#include <utility>
#include <functional>
#include <iostream>

using namespace std;

vector<int> dijkstra(int V, vector<vector<int>> &edges, int src) {

    // Create adjacency list
    vector<vector<pair<int, int>>> adj(V);

    for (auto edge : edges) {
        int u = edge[0];
        int v = edge[1];
        int weight = edge[2];

        adj[u].push_back({v, weight});
        adj[v].push_back({u, weight});  // Only for UNDIRECTED graph
    }

    // Priority queue: {distance, node}
    priority_queue<
        pair<int, int>,
        vector<pair<int, int>>,
        greater<pair<int, int>>
    > pq;

    // Distance array
    vector<int> dist(V, 1e9);

    dist[src] = 0;
    pq.push({0, src});

    while (!pq.empty()) {

        int distance = pq.top().first;
        int node = pq.top().second;

        pq.pop();

        for (auto it : adj[node]) {

            int adjNode = it.first;
            int edgeWeight = it.second;

            if (distance + edgeWeight < dist[adjNode]) {

                dist[adjNode] = distance + edgeWeight;

                pq.push({dist[adjNode], adjNode});
            }
        }
    }

    return dist;
}