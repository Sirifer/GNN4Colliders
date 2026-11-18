#include <TFile.h>
#include <TObject.h>
#include <TTree.h>
#include <iostream>

int append_root_tree(const char *target_path, const char *source_path, const char *tree_name, bool overwrite = false) {
    TFile source(source_path, "READ");
    if (source.IsZombie()) {
        std::cerr << "Cannot open temporary source: " << source_path << std::endl;
        return 10;
    }
    TTree *source_tree = nullptr;
    source.GetObject(tree_name, source_tree);
    if (source_tree == nullptr) {
        std::cerr << "Tree not found in temporary source: " << tree_name << std::endl;
        return 11;
    }

    TFile target(target_path, "UPDATE");
    if (target.IsZombie()) {
        std::cerr << "Cannot open target in UPDATE mode: " << target_path << std::endl;
        return 12;
    }
    if (target.Get(tree_name) != nullptr && !overwrite) {
        std::cerr << "Target tree already exists: " << tree_name << std::endl;
        return 13;
    }

    target.cd();
    TTree *cloned = source_tree->CloneTree(-1, "fast");
    if (cloned == nullptr) {
        std::cerr << "CloneTree failed for: " << tree_name << std::endl;
        return 14;
    }
    cloned->SetName(tree_name);
    cloned->SetTitle("Merged SR and Sideband categories with source tree tag");
    const Long64_t entries = cloned->GetEntries();
    const int bytes = cloned->Write(tree_name, overwrite ? TObject::kOverwrite : 0);
    target.Close();
    source.Close();
    if (bytes <= 0) {
        std::cerr << "Write failed for: " << tree_name << std::endl;
        return 15;
    }
    std::cout << "Appended " << tree_name << " with " << entries << " entries" << std::endl;
    return 0;
}
