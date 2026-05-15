from core.memory import MemoryNode

def test():
    memory = MemoryNode()
    
    # 1. Store test events
    memory.store("User requested a soil scan. NPK levels were low.")
    memory.store("Deployed 50kg/ha of Urea fertilizer to the north field.")
    memory.store("Detected Leaf Blight on the tomato crop. Quarantined area.")
    
    # 2. Test semantic recall (Notice the query uses different words than the stored text)
    print("\n--- Recall Test ---")
    results = memory.recall("Did we apply any chemicals recently?")
    
    for res in results:
        print(res)

if __name__ == "__main__":
    test()