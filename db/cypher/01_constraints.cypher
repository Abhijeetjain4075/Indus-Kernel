// Neo4j schema initialisation for the Indus Kernel.
// See ARCHITECTURE.md Section 7.4 for the full Cypher.
//
// This file is mounted into the Neo4j container at /var/lib/neo4j/import.
// Run via:
//   docker compose exec neo4j cypher-shell -u neo4j -p indus < /var/lib/neo4j/import/01_constraints.cypher
//
// Or from Python:
//   from ik_memory.adapters.neo4j import run_cypher_file
//   run_cypher_file("db/cypher/01_constraints.cypher")

// Constraints
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT relation_id IF NOT EXISTS FOR ()-[r:RELATION]-() REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT memory_node_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT community_id IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT skill_id IF NOT EXISTS FOR (s:Skill) REQUIRE s.id IS UNIQUE;

// Indexes
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_tenant IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id);
CREATE INDEX memory_scope IF NOT EXISTS FOR (m:Memory) ON (m.scope);
CREATE INDEX memory_tenant IF NOT EXISTS FOR (m:Memory) ON (m.tenant_id);
CREATE INDEX community_level IF NOT EXISTS FOR (c:Community) ON (c.level);
CREATE INDEX skill_category IF NOT EXISTS FOR (s:Skill) ON (s.category);
